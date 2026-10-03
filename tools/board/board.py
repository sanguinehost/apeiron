#!/usr/bin/env python3
"""Operately board driver for the Sanguine stream (space ``sanguine``).

The board tracks status; it does not replace the epic/wave/task decomposition, a
gate or a control (``AGENTS.md`` §9.3). This tool makes the board *drivable*: a
session reads ``next`` and starts, a wave boundary reads ``--check`` and stops on
drift, and every reconciliation leaves its audit trail on the board.

Protocol
========

**One writer per row.** The lead creates, closes, re-dates and re-describes. A
worker moves only its own task's status, and read-confirms: a transport error is
not evidence of refusal, so ``set-status`` always reads back and the read-back is
the verdict (this instance answers ``500`` for a mutation that landed).

**A dispatch names its row.** Every row's description carries its own labelled
field, ``board: <row>`` naming the row the dispatch moves (the task id or the plan
row it serves), or ``board: none -- internal lane`` for a lane with no board row.
A row without the declaration is a finding: a dispatch no board row can be traced
from is not drivable. A fold without its board move is drift.

**The reconciliation check is the red.** ``--check`` compares the board's
non-closed tasks against the repository plan's non-closed rows and classifies
*every* row as ``mirrored``, ``internal-by-design`` (named, with the naming in the
description) or ``missing``. Drift is a named line and a nonzero exit. An
unattributed row is checked too: it is outside the project selection, which is
why it drifts.

**The check-in is the audit trail.** ``post`` writes the reconciliation record to
the project and read-confirms that it landed.

**Due dates only from dated facts.** The tool never writes a date and never infers
one. A ``due:`` field in a description is a claim about the world, so it is
reported unless the row carries a dated fact -- a receipt or verdict reference, a
sha256 or a revision.

Modes
=====

``list``     status, milestone, the **gate** and the **source path(s)** parsed from
             the description, id, name.
``next``     the highest-priority actionable task (status ``pending`` /
             ``in_progress``), ordered by milestone then id, printed with its full
             id, gate and sources so a session can start from it.
``check``    reconcile mode; nonzero when the board cannot drive or has drifted.
``post``     write the reconciliation record as a project check-in, read-confirmed.
``set-status``  move one task's status, read-confirmed. Dry-run without ``--apply``.

Sources of truth
================

Named explicitly in every ``check``/``post`` output with the sha256 of the file
read, so a finding is bound to the plan revision it was computed against:

* apeiron -- ``docs/plan.md`` (epics/waves/tasks tables)
* song -- ``plan/ledger.tsv`` and ``plan/fleet.tsv``

``--plan PATH`` (repeatable) names them; a relative path resolves against
``--repo-root``. A board document may also carry its plan inline, which is how
the offline fixtures work: no key, no network, no repository read. ``--fixtures``
makes a run hermetic -- the inline plan is the source of truth.

Usage
=====

::

    python3 tools/board/board.py --check --controls
    python3 tools/board/board.py list --live
    python3 tools/board/board.py next --live --project apeiron
    python3 tools/board/board.py --check --live --project apeiron-ExNAtv3hObMmq01Ab20DfA
    python3 tools/board/board.py post --live --project apeiron --apply
    python3 tools/board/board.py set-status <id>=Done --live --apply
    python3 tools/board/board.py --rules
    python3 -m unittest discover -s tools/board -p 'test_*.py'

Exit status: ``0`` success; ``1`` a finding (rule or drift); ``2`` usage,
transport or control-suite failure; ``3`` ``next`` found no actionable task.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable

SCHEMA = 1
DEFAULT_SPACE_ID = "sanguine-COpRXpkvCi19pKoDZnOPg0"
DEFAULT_BASE_URL = "https://projects.tas.twn.network/api/external/v1"
DEFAULT_PROJECTS = ("apeiron", "song", "penglai")
# The declared source of truth per project (AGENTS.md §9.3 and song's own plan
# arm). Relative paths resolve against --repo-root; a path absent there is not
# read, and the omission is reported rather than guessed.
PROJECT_PLANS = {
    "apeiron": ("docs/plan.md",),
    "song": ("plan/ledger.tsv", "plan/fleet.tsv"),
    "penglai": (),
}
KEY_ENV = "OPERATELY_API_KEY"
HTTP_TIMEOUT_S = 20
# Below this length a value is not a credential, and matching it would corrupt the
# text it appears in rather than redact a secret.
MIN_SCRUB_LENGTH = 8

# ---------------------------------------------------------------------------
# Rule names. Stable interface: the controls assert on them by name, so a rename
# breaks a control instead of silently weakening it.
# ---------------------------------------------------------------------------

# Board-contract rules (a task that cannot be started from the board).
RULE_MILESTONE_EMPTY = "milestone-without-tasks"
RULE_NO_PATHS = "task-without-source-paths"
RULE_NO_GATE = "task-without-gate"
RULE_DONE_NO_RECEIPT = "done-without-receipt"
RULE_WRITE_NOT_LANDED = "write-did-not-land"
# Extension rules, same failure class: the board cannot drive.
RULE_UNKNOWN_MILESTONE = "task-with-unknown-milestone"
RULE_UNATTRIBUTED = "unattributed-task"
RULE_PROJECT_MISSING = "board-project-missing"
# Reconciliation rules (drift against the named source of truth).
RULE_PLAN_UNDECLARED = "plan-source-undeclared"
RULE_DRIFT_PLAN_ROW = "drift-plan-row-not-on-board"
RULE_DRIFT_BOARD_ROW = "drift-board-row-not-in-plan"
RULE_DRIFT_STALE = "drift-board-open-against-closed-plan"
# Protocol rules: a dispatch names its row, and a due date names a dated fact.
RULE_NO_BOARD_DECLARATION = "task-without-board-declaration"
RULE_DUE_WITHOUT_FACT = "due-date-without-dated-fact"

CONTRACT_RULES = (
    RULE_MILESTONE_EMPTY,
    RULE_NO_PATHS,
    RULE_NO_GATE,
    RULE_DONE_NO_RECEIPT,
    RULE_WRITE_NOT_LANDED,
)
PROTOCOL_RULES = (RULE_NO_BOARD_DECLARATION, RULE_DUE_WITHOUT_FACT)
RECONCILE_RULES = (RULE_PLAN_UNDECLARED, RULE_DRIFT_PLAN_ROW, RULE_DRIFT_BOARD_ROW, RULE_DRIFT_STALE)
EXTENSION_RULES = (RULE_UNKNOWN_MILESTONE, RULE_UNATTRIBUTED, RULE_PROJECT_MISSING)
ALL_RULES = CONTRACT_RULES + PROTOCOL_RULES + RECONCILE_RULES + EXTENSION_RULES

# ---------------------------------------------------------------------------
# Description parsing
# ---------------------------------------------------------------------------

BACKTICK = re.compile(r"`([^`\n]+)`")
GATE_HEADS = frozenset(
    {
        "cargo",
        "lake",
        "lean",
        "python",
        "python3",
        "pytest",
        "bash",
        "sh",
        "make",
        "ninja",
        "perf",
        "timeout",
        "systemd-run",
        "flock",
        "time",
        "tools/gate",
        "tools/gate/target/debug/gate",
        "proof/regen.sh",
        "proof/model-patches.sh",
        "proof/install-patches.sh",
    }
)
PATH_EXTENSIONS = (
    ".rs",
    ".lean",
    ".toml",
    ".md",
    ".sh",
    ".py",
    ".json",
    ".tsv",
    ".csv",
    ".c",
    ".h",
    ".slang",
    ".yml",
    ".yaml",
    ".txt",
    ".lock",
)
RECEIPT_PREFIXES = ("docs/evidence/", "docs/receipts/", "receipt", "artifacts/")
URL = re.compile(r"https?://\S+")

LABEL_GATE = re.compile(r"\b(?:gate|gates|acceptance)\s*[:：]\s*(.+)", re.I)
LABEL_PATHS = re.compile(
    r"\b(?:source|sources|path|paths|file|files|owned|owns|reads)\s*[:：]\s*(.+)", re.I
)
LABEL_RECEIPT = re.compile(
    r"\b(?:receipt|receipts|verdict|verdicts|evidence|control)\s*[:：]\s*(.+)", re.I
)
PROJECT_MENTION = re.compile(r"\bproject\s+[`\"']?([A-Za-z][A-Za-z0-9_.-]{1,40})[`\"']?", re.I)
# A task naming itself as an internal lane, which carries no board row by design.
INTERNAL_MARKERS = ("internal-by-design", "internal by design", "internal lane", "internal:")
# A dispatch names the row it moves. The field is a labelled line of its own, so
# a mention in prose ("moved onto the board: three rows") is not a declaration.
BOARD_DECLARATION = re.compile(r"(?:^|\n)[ \t>*_-]*board\s*[:：]\s*([^\n]+)", re.I)
DUE_FIELD = re.compile(r"(?:^|\n)[ \t>*_-]*due(?:\s+date)?\s*[:：]\s*([^\n]+)", re.I)
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NO_ROW = re.compile(r"\b(?:none|n/?a|no row|not applicable|internal)\b", re.I)
# A sha256 prefix or a git revision: an anchor a date can be bound to.
DIGEST = re.compile(r"\b[0-9a-f]{16,64}\b")


def flatten_description(raw: Any) -> str:
    """Flatten a task description to plain text.

    The instance stores descriptions as a ProseMirror JSON document in a JSON
    string; fixtures and hand-written updates use markdown. Both are accepted and
    ``None`` is the empty string.
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = json.dumps(raw)
    text = raw
    stripped = raw.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            doc = json.loads(raw)
        except (ValueError, TypeError):
            doc = None
        if isinstance(doc, dict) and ("content" in doc or doc.get("type") == "doc"):
            text = _prosemirror(doc)
    return text


def _prosemirror(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    kind = node.get("type")
    if kind == "text":
        return str(node.get("text") or "")
    if kind in ("doc", "blockquote", "bullet_list", "ordered_list", "list_item"):
        return "".join(_prosemirror(child) + "\n" for child in node.get("content") or [])
    if kind in ("paragraph", "heading", "code_block"):
        return "".join(_prosemirror(child) for child in node.get("content") or []) + "\n"
    return "".join(_prosemirror(child) for child in node.get("content") or [])


def _tokens(value: str) -> list[str]:
    spans = [t.strip() for t in BACKTICK.findall(value)]
    if spans:
        return spans
    return [w for w in value.strip().split() if w]


def _labelled(text: str, pattern: re.Pattern[str]) -> list[str]:
    out: list[str] = []
    for match in pattern.finditer(text):
        for line in match.group(1).splitlines():
            line = line.strip()
            if line:
                out.extend(_tokens(line))
    return out


def _is_path(token: str) -> bool:
    """A source or governing file. A repo-root file (``README.md``) counts too."""
    if not token or any(ch.isspace() for ch in token):
        return False
    return token.endswith(PATH_EXTENSIONS)


def _is_gate(token: str) -> bool:
    """A gate is a *runnable* command, not a path that happens to be a script.

    An invocation carries an argument (``cargo test``, ``proof/regen.sh --check``,
    ``python3 tools/rooting/check.py --check``) or names an extensionless gate
    binary under ``tools/`` or ``proof/`` (``tools/gate``).
    """
    token = token.strip()
    if token.startswith("$ "):
        token = token[2:].strip()
    words = token.split()
    while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=\S*", words[0]):
        words = words[1:]
    if not words:
        return False
    if len(words) > 1:
        return words[0] in GATE_HEADS
    if "/" in words[0] and "." not in words[0].rsplit("/", 1)[-1]:
        return words[0].startswith(("tools/", "proof/"))
    return False


def _split(tokens: Iterable[str]) -> tuple[list[str], list[str]]:
    gates: list[str] = []
    paths: list[str] = []
    for token in tokens:
        if _is_gate(token):
            gates.append(token)
        elif _is_path(token):
            paths.append(token)
    return _dedupe(gates), _dedupe(paths)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def gates_of(text: str) -> list[str]:
    """Every gate the description names, from labelled fields then prose."""
    labelled, _ = _split(_labelled(text, LABEL_GATE))
    spans = [t.strip() for t in BACKTICK.findall(text)]
    inline, _ = _split(spans)
    return _dedupe(labelled + inline)


def paths_of(text: str) -> list[str]:
    """Every source path the description names, from labelled fields then prose."""
    labelled, _ = _split(_labelled(text, LABEL_PATHS))
    _, inline = _split([t.strip() for t in BACKTICK.findall(text)])
    return _dedupe(labelled + inline)


def receipts_of(text: str, comments: Iterable[str] = ()) -> list[str]:
    """Receipt/verdict references: a labelled field, an evidence path or a URL."""
    found: list[str] = []
    for candidate in _labelled(text, LABEL_RECEIPT):
        if _is_path(candidate) or URL.search(candidate):
            found.append(candidate)
    for source in (text, *comments):
        for span in BACKTICK.findall(source):
            span = span.strip()
            if any(span.lower().startswith(p) for p in RECEIPT_PREFIXES) or URL.search(span):
                found.append(span)
        for url in URL.findall(source):
            found.append(url.rstrip(".,);"))
    return _dedupe(found)


def project_mentions(text: str) -> list[str]:
    return _dedupe(PROJECT_MENTION.findall(text))


def internal_named_by(text: str) -> str:
    """The internal-by-design marker a description carries, or ``""``."""
    lowered = text.lower()
    for marker in INTERNAL_MARKERS:
        if marker in lowered:
            return marker
    return ""


def board_declaration(text: str) -> tuple[str, str]:
    """The ``board:`` field a dispatch must carry.

    ``board: <row>`` names the row the dispatch moves -- the task's own id or the
    plan row it serves. ``board: none -- internal lane`` declares a lane that has
    no board row. Returns ``(kind, value)`` with kind ``row``, ``internal``, or
    ``""`` when the field is absent: a row without the declaration is a finding,
    because a dispatch no board row can be found from is not drivable.
    """
    for match in BOARD_DECLARATION.finditer(text):
        value = re.sub(r"\s+", " ", match.group(1).replace("`", " ").strip()).strip()
        if not value:
            continue
        first = value.split(" ", 1)[0]
        rest = value[len(first) :].strip()
        # A placeholder such as ``board: <row-id>`` documents the syntax; it
        # declares nothing. Angle brackets cannot occur in a real row token, so
        # a file that teaches the declaration grammar does not trip a check on
        # its own example text.
        if "<" in first or ">" in first:
            continue
        if NO_ROW.match(first):
            return "internal", value
        # A row token is a single identifier. Prose that happens to contain
        # ``Board:`` (a document describing the board, an ``- Board: company
        # TWN, ...`` line) is not a declaration.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", first.rstrip(".,;:")):
            continue
        # A declaration's remainder is punctuation or a ``-- reason``; a sentence
        # after the token (``- Board: company TWN, space ...``) is prose about
        # the board, not a declaration.
        if rest.strip("`.,;:()[] ") and not rest.startswith("--"):
            continue
        return "row", value
    return "", ""


def declared_row(value: str) -> str:
    """The row token a ``board:`` declaration names, or ``""``."""
    tokens = value.split()
    return re.sub(r"[`.,;:]+$", "", re.sub(r"^[`.,;:]+", "", tokens[0])) if tokens else ""


def due_date_of(text: str) -> str:
    """The ISO date a ``due:`` field claims, or ``""``."""
    match = DUE_FIELD.search(text)
    if not match:
        return ""
    found = ISO_DATE.search(match.group(1))
    return found.group(0) if found else ""


def dated_facts(text: str, receipts: Iterable[str] = ()) -> list[str]:
    """What can back a date: a receipt/verdict reference, a digest or a revision.

    A date is a commitment; only a recorded fact supports one. The lead owns
    re-dating, and the tool never writes a date -- it only reports a date with
    nothing behind it.
    """
    found = list(receipts)
    found.extend(match.group(0) for match in DIGEST.finditer(text))
    return _dedupe(found)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


# ---------------------------------------------------------------------------
# Plan parsing: the source of truth, named and digested
# ---------------------------------------------------------------------------

PLAN_ID_IN_CODE = re.compile(r"`([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)`")
# A slug shorter than this is too weak to match a plan row by name.
MIN_SLUG_MATCH = 8
CLOSED_MARKERS = (
    "landed",
    "closed",
    "retired",
    "done",
    "complete",
    "completed",
    "historical",
    "superseded",
    "returned",
)
TASK_COLUMNS = ("task", "tasks")
TASK_TABLE_KEYWORDS = ("wave", "task", "milestone", "state", "horizon")


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_closed(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{marker}\b", lowered) for marker in CLOSED_MARKERS)


def _read_table(lines: list[str], start: int) -> tuple[list[str], list[list[str]], int]:
    """Read one markdown table. Returns (header cells, body rows, next index)."""
    header = [c.strip() for c in lines[start].strip().strip("|").split("|")]
    index = start + 1
    body: list[list[str]] = []
    if index < len(lines) and re.fullmatch(r"\|[\s:|-]+\|", lines[index].strip()):
        index += 1
    while index < len(lines) and lines[index].strip().startswith("|"):
        cells = [c.strip() for c in lines[index].strip().strip("|").split("|")]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        body.append(cells)
        index += 1
    return header, body, index


def parse_markdown_plan(text: str, source: str) -> list[dict[str, Any]]:
    """Rows of every plan table in a markdown plan.

    A row is identified by the task id it carries in backticks
    (``APEIRON-E6-W4-T1``), or, when the row carries none, by its section and
    wave. Closed-ness is the row's own wording: a row that says ``landed``,
    ``closed``, ``retired``, ``done``, ``complete``, ``historical``,
    ``superseded`` or ``returned`` is closed, and every other row is non-closed.
    This is a text classification over a prose document, not a verified parse:
    every classification is printed by ``--check`` so a reader can audit it.
    """
    rows: list[dict[str, Any]] = []
    lines = text.splitlines()
    section = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        if line.strip().startswith("|"):
            header, body, index = _read_table(lines, index)
            columns = [c.lower().strip() for c in header]
            if any(word in " ".join(columns) for word in TASK_TABLE_KEYWORDS):
                rows.extend(_markdown_rows(header, columns, body, section, source))
            continue
        index += 1
    return rows


def _markdown_rows(
    header: list[str],
    columns: list[str],
    body: list[list[str]],
    section: str,
    source: str,
) -> list[dict[str, Any]]:
    def column(*names: str) -> int | None:
        for name in names:
            if name in columns:
                return columns.index(name)
        return None

    task_at = column(*TASK_COLUMNS)
    if task_at is None:
        return []
    wave_at = column("wave", "milestone")
    gate_at = column("gate", "gates")
    files_at = column("files", "file", "paths", "owned")
    state_at = column("state", "status")
    section_slug = slug(section) or "plan"
    rows: list[dict[str, Any]] = []
    for cells in body:
        row_text = " ".join(cells)
        if not row_text.strip(" |"):
            continue
        ids = _dedupe(PLAN_ID_IN_CODE.findall(row_text))
        task_cell = cells[task_at] if task_at < len(cells) else ""
        if ids:
            row_id = ids[0]
            also = ids[1:]
        else:
            wave = cells[wave_at] if wave_at is not None and wave_at < len(cells) else ""
            row_id = f"{section_slug}/{slug(wave) or slug(task_cell)[:40]}"
            also = []
        name = re.sub(r"`[^`]*`", "", task_cell).strip(" —-*:")
        name = re.sub(r"\s+", " ", name)[:120]
        closed_by = ""
        if state_at is not None and state_at < len(cells) and cells[state_at]:
            if _row_closed(cells[state_at]):
                closed_by = cells[state_at].strip()
        if not closed_by and _row_closed(row_text):
            closed_by = "row wording"
        gate_text = cells[gate_at] if gate_at is not None and gate_at < len(cells) else ""
        files_text = cells[files_at] if files_at is not None and files_at < len(cells) else ""
        for row_ident in [row_id, *also]:
            rows.append(
                {
                    "id": row_ident,
                    "name": name or row_ident,
                    "section": section,
                    "source": source,
                    "closed": bool(closed_by),
                    "closed_by": closed_by,
                    "gates": gates_of(gate_text),
                    "paths": paths_of(files_text),
                }
            )
    return rows


def parse_tsv_plan(text: str, source: str) -> list[dict[str, Any]]:
    """Rows of a tab-separated ledger or fleet file (song's sources of truth)."""
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if not reader.fieldnames or "id" not in [f.strip() for f in reader.fieldnames]:
        return rows
    for raw in reader:
        clean = {(k or "").strip(): (v or "") for k, v in raw.items()}
        row_id = (clean.get("id") or "").strip()
        if not row_id:
            continue
        state = (clean.get("state") or clean.get("status") or "").strip()
        name = (
            clean.get("task")
            or clean.get("name")
            or clean.get("lane")
            or clean.get("residual")
            or row_id
        ).strip()
        rows.append(
            {
                "id": row_id,
                "name": name[:120],
                "section": source,
                "source": source,
                "closed": _row_closed(state) or state.lower() in {"closed", "done"},
                "closed_by": state,
                "gates": gates_of(clean.get("gate") or ""),
                "paths": _dedupe(
                    [p.strip() for p in re.split(r"[;,]", clean.get("paths") or "") if p.strip()]
                ),
            }
        )
    return rows


def parse_plan_source(path: str, text: str) -> list[dict[str, Any]]:
    if path.endswith(".tsv") or path.endswith(".csv"):
        delimiter = "\t" if path.endswith(".tsv") else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if reader.fieldnames and "id" in [f.strip() for f in reader.fieldnames]:
            return parse_tsv_plan(text, path)
    return parse_markdown_plan(text, path)


def load_plan(paths: Iterable[str], repo_root: str) -> dict[str, Any]:
    """Read the named sources of truth and digest each file."""
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path
        if not os.path.isabs(path):
            path = os.path.join(repo_root, raw_path)
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise TransportError(f"plan source unreadable: {exc}") from None
        parsed = parse_plan_source(raw_path, text)
        sources.append(
            {
                "path": raw_path,
                "sha256": sha256_of(text),
                "rows": len(parsed),
                "open": sum(1 for r in parsed if not r["closed"]),
            }
        )
        rows.extend(parsed)
    return {"sources": sources, "rows": rows}


def plan_from_document(doc: dict[str, Any], project: str | None = None) -> dict[str, Any] | None:
    """The plan carried inline by a board document (how the fixtures work).

    When a project is named and the sources declare one, only that project's
    sources are read: song's ledger is not evidence about apeiron's plan.
    """
    plan = doc.get("plan")
    if not isinstance(plan, dict):
        return None
    entries = plan.get("sources") or []
    if project and any(e.get("project") for e in entries):
        lowered = project.lower()
        entries = [
            e for e in entries if not e.get("project") or str(e["project"]).lower() == lowered
        ]
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        text = entry.get("text") or ""
        parsed = parse_plan_source(entry.get("path") or "inline", text)
        sources.append(
            {
                "path": entry.get("path") or "inline",
                "sha256": entry.get("sha256") or sha256_of(text),
                "rows": len(parsed),
                "open": sum(1 for r in parsed if not r["closed"]),
            }
        )
        rows.extend(parsed)
    return {"sources": sources, "rows": rows}


# ---------------------------------------------------------------------------
# Normalized board document
# ---------------------------------------------------------------------------


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_task(raw: dict[str, Any], milestone_index: dict[str, int]) -> dict[str, Any]:
    status = raw.get("status") or {}
    if isinstance(status, str):
        status = {"value": status, "label": status}
    milestone = raw.get("milestone")
    if isinstance(milestone, str):
        milestone = {"id": milestone, "title": milestone}
    project = raw.get("project")
    if isinstance(project, str):
        project = {"id": project, "name": project}
    text = flatten_description(raw.get("description"))
    comments = [flatten_description(c) for c in (raw.get("comments") or []) if isinstance(c, str)]
    milestone_id = (milestone or {}).get("id")
    return {
        "id": raw.get("id") or "",
        "name": raw.get("name") or "",
        "status_value": status.get("value") or "",
        "status_label": status.get("label") or status.get("value") or "",
        "milestone_id": milestone_id,
        "milestone_title": (milestone or {}).get("title"),
        "milestone_index": milestone_index.get(milestone_id) if milestone_id else None,
        "project_id": (project or {}).get("id"),
        "project_name": (project or {}).get("name"),
        "text": text,
        "gates": gates_of(text),
        "paths": paths_of(text),
        "receipts": receipts_of(text, comments),
        "plan_ids": _dedupe(PLAN_ID_IN_CODE.findall(text)),
        "internal_by": internal_named_by(text),
    }


def normalize_document(raw: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(raw, dict), "board document must be an object")
    _require(raw.get("schema") == SCHEMA, f"board document schema must be {SCHEMA}")
    space = raw.get("space") or {}
    _require(isinstance(space, dict), "space must be an object")
    milestones_raw = raw.get("milestones") or []
    _require(isinstance(milestones_raw, list), "milestones must be a list")
    milestones: list[dict[str, Any]] = []
    milestone_index: dict[str, int] = {}
    for position, item in enumerate(milestones_raw):
        if isinstance(item, str):
            item = {"id": item, "title": item}
        _require(isinstance(item, dict) and item.get("id"), "milestone needs an id")
        project = item.get("project")
        if isinstance(project, str):
            project = {"id": project, "name": project}
        milestones.append(
            {
                "id": item["id"],
                "title": item.get("title") or item["id"],
                "status": item.get("status") or "",
                "project_id": (project or {}).get("id"),
                "project_name": (project or {}).get("name"),
                "index": position,
            }
        )
        milestone_index[item["id"]] = position
    projects: list[dict[str, Any]] = []
    for item in raw.get("projects") or []:
        if isinstance(item, str):
            item = {"id": item, "name": item}
        projects.append({"id": item.get("id"), "name": item.get("name") or item.get("id")})
    tasks_raw = raw.get("tasks") or []
    _require(isinstance(tasks_raw, list), "tasks must be a list")
    tasks = [normalize_task(t, milestone_index) for t in tasks_raw]
    # A task carries its milestone id, not always its title; the milestone
    # inventory is the authority for the title, so it is filled in here rather
    # than being dropped from every rendered row.
    titles = {m["id"]: m["title"] for m in milestones}
    for task in tasks:
        if not task["milestone_title"] and task["milestone_id"]:
            task["milestone_title"] = titles.get(task["milestone_id"])
    writes = []
    for entry in raw.get("writes") or []:
        _require(isinstance(entry, dict), "writes entries must be objects")
        writes.append(
            {
                "task_id": entry.get("task_id") or "",
                "target_status": entry.get("target_status") or "",
                "observed_status": entry.get("observed_status") or "",
                "transport": entry.get("transport") or "",
            }
        )
    document = {
        "schema": SCHEMA,
        "origin": raw.get("origin") or "",
        "space": {"id": space.get("id") or "", "name": space.get("name") or ""},
        "projects": projects,
        "milestones": milestones,
        "tasks": tasks,
        "writes": writes,
        "notes": [n for n in (raw.get("notes") or []) if isinstance(n, str)],
        "plan": raw.get("plan") if isinstance(raw.get("plan"), dict) else None,
        "internal": [str(i) for i in (raw.get("internal") or [])],
    }
    return document


def attribute_projects(doc: dict[str, Any]) -> None:
    """Resolve the project of a task that carries none.

    A task with no project object belongs to the space. When the space holds
    projects, a task naming one in prose ("moved into project `song`") is
    attributed to it; a task naming none stays unattributed and is reported by
    ``unattributed-task`` -- no project-scoped driver can select it.
    """
    names = {p["name"]: p["id"] for p in doc["projects"] if p.get("name")}
    for task in doc["tasks"]:
        if task["project_id"]:
            continue
        for mention in project_mentions(task["text"]):
            for name, pid in names.items():
                if mention.lower() == name.lower():
                    task["project_id"] = pid
                    task["project_name"] = name
                    break
            if task["project_id"]:
                break


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

ACTIONABLE = ("pending", "in_progress")
DONE_VALUES = ("done", "complete", "completed", "closed")
CANCELED_VALUES = ("canceled", "cancelled", "dropped", "archived")


def status_bucket(task: dict[str, Any]) -> str:
    value = (task["status_value"] or "").lower()
    if value in ACTIONABLE:
        return "actionable"
    if value in DONE_VALUES:
        return "done"
    if value in CANCELED_VALUES:
        return "canceled"
    return "other"


def non_closed_tasks(doc: dict[str, Any], project: str | None = None) -> list[dict[str, Any]]:
    return [t for t in select_tasks(doc, project) if status_bucket(t) != "done"]


def select_tasks(
    doc: dict[str, Any], project: str | None = None, include_canceled: bool = False
) -> list[dict[str, Any]]:
    tasks = doc["tasks"]
    if project:
        lowered = project.lower()
        tasks = [
            t
            for t in tasks
            if (t["project_name"] or "").lower() == lowered
            or (t["project_id"] or "").lower() == lowered
            or (t["id"] or "").lower().startswith(lowered + "-")
        ]
    if not include_canceled:
        tasks = [t for t in tasks if status_bucket(t) != "canceled"]
    return tasks


def sort_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Milestone order, then id. Unassigned last: an unplaced task has no order."""

    def key(task: dict[str, Any]) -> tuple[int, int, str]:
        index = task["milestone_index"]
        return (0, index, task["id"]) if index is not None else (1, 0, task["id"])

    return sorted(tasks, key=key)


def next_task(doc: dict[str, Any], project: str | None = None) -> dict[str, Any] | None:
    candidates = [t for t in select_tasks(doc, project) if status_bucket(t) == "actionable"]
    ordered = sort_tasks(candidates)
    return ordered[0] if ordered else None


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


class Finding(dict):
    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self['rule']}: {self['detail']}"


def _finding(rule: str, detail: str, **extra: Any) -> Finding:
    item = Finding(rule=rule, detail=detail)
    item.update(extra)
    return item


def reconcile(
    doc: dict[str, Any],
    plan: dict[str, Any] | None,
    project: str | None = None,
    internal: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[Finding]]:
    """Classify every non-closed row against the named source of truth.

    Board row -> plan row by the plan id the description names, else by slug.
    Plan row -> board row by any board row naming it or carrying its slug.
    A board row that matches nothing is ``internal-by-design`` only when its own
    description names that; otherwise it is drift. Every classification is
    returned, mirrored and internal included, so the output is the audit trail.
    """
    board_rows = non_closed_tasks(doc, project)
    plan_rows = list(plan["rows"]) if plan else []
    plan_open = [r for r in plan_rows if not r["closed"]]
    plan_closed_ids = {r["id"] for r in plan_rows if r["closed"]}
    plan_by_slug: dict[str, dict[str, Any]] = {}
    for row in plan_rows:
        if row.get("name"):
            plan_by_slug.setdefault(slug(row["name"]), row)
    named_internal = {i.lower() for i in internal} | {i.lower() for i in doc.get("internal") or []}

    def match_by_slug(name: str) -> dict[str, Any] | None:
        """A plan row's name is prose, so matching is on the slug prefix."""
        target = slug(name)
        if len(target) < MIN_SLUG_MATCH:
            return None
        candidates = [
            row
            for key, row in plan_by_slug.items()
            if len(key) >= MIN_SLUG_MATCH and (key.startswith(target) or target.startswith(key))
        ]
        return max(candidates, key=lambda r: len(slug(r["name"]))) if candidates else None

    classifications: list[dict[str, Any]] = []
    findings: list[Finding] = []
    matched_plan_ids: set[str] = set()

    if plan is None:
        return classifications, [
            _finding(
                RULE_PLAN_UNDECLARED,
                "no plan source of truth was declared; pass --plan PATH or carry a "
                "plan block in the board document, so every classification is bound "
                "to a named file and its sha256",
            )
        ]

    for task in board_rows:
        matched = None
        for plan_id in task["plan_ids"]:
            candidate = next((r for r in plan_rows if r["id"] == plan_id), None)
            if candidate is not None:
                matched = candidate
                break
        if matched is None:
            matched = match_by_slug(task["name"])
        if matched is not None:
            matched_plan_ids.add(matched["id"])
            classification = "mirrored"
            detail = f"plan row {matched['id']} ({matched['name']})"
            if matched["closed"]:
                classification = "stale"
                findings.append(
                    _finding(
                        RULE_DRIFT_STALE,
                        f"board task {task['id']} ({task['name']}) is non-closed "
                        f"while plan row {matched['id']} is closed ({matched['closed_by'] or 'row wording'})",
                        task=task["id"],
                        plan=matched["id"],
                    )
                )
                detail = f"plan row {matched['id']} is CLOSED ({matched['closed_by']})"
            classifications.append(
                {"class": classification, "board": task["id"], "plan": matched["id"], "detail": detail}
            )
            continue
        named = task["internal_by"] or (
            task["name"] if task["name"].lower() in named_internal else ""
        )
        if task["id"].lower() in named_internal:
            named = "named in the document's internal list"
        if named:
            classifications.append(
                {
                    "class": "internal-by-design",
                    "board": task["id"],
                    "plan": "",
                    "detail": f"named by {named!r}",
                }
            )
            continue
        classifications.append({"class": "missing", "board": task["id"], "plan": "", "detail": ""})
        findings.append(
            _finding(
                RULE_DRIFT_BOARD_ROW,
                f"board task {task['id']} ({task['name']}) is non-closed and names no "
                f"plan row; it is not named internal-by-design either "
                f"(plan ids seen: {', '.join(task['plan_ids']) or 'none'})",
                task=task["id"],
            )
        )

    # A plan row can also be mirrored by name rather than by the id its own row
    # carries. Resolve every board row that did not match by id once: the
    # comprehension below used to re-derive it while shadowing ``row``, which
    # made the loop's own variable unreachable.
    slug_matched_ids: set[str] = set()
    for task in board_rows:
        if any(any(r["id"] == plan_id for r in plan_rows) for plan_id in task["plan_ids"]):
            continue
        candidate = match_by_slug(task["name"])
        if candidate is not None:
            slug_matched_ids.add(candidate["id"])

    for row in plan_open:
        if row["id"] in matched_plan_ids or row["id"] in slug_matched_ids:
            continue
        classifications.append({"class": "missing", "board": "", "plan": row["id"], "detail": ""})
        findings.append(
            _finding(
                RULE_DRIFT_PLAN_ROW,
                f"plan row {row['id']} ({row['name']}, {row['source']}) is non-closed "
                "and has no non-closed board row",
                plan=row["id"],
            )
        )
    return classifications, findings


def check(
    doc: dict[str, Any],
    project: str | None = None,
    require_attribution: bool = True,
    plan: dict[str, Any] | None = None,
    reconcile_findings: list[Finding] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    tasks = select_tasks(doc, project)
    # A row with no project is outside a project-scoped selection, which is
    # exactly why it drifts. Its own contract and protocol state is checked too,
    # so a space-level row is never passed over because it is unattributed.
    scope = list(tasks)
    seen = {t["id"] for t in tasks}
    for task in doc["tasks"]:
        if task["id"] in seen or task["project_id"]:
            continue
        if status_bucket(task) == "canceled":
            continue
        scope.append(task)
    checked = {t["id"] for t in scope}

    for milestone in doc["milestones"]:
        if project and (milestone["project_name"] or "").lower() != project.lower():
            continue
        if not any(t["milestone_id"] == milestone["id"] for t in doc["tasks"]):
            findings.append(
                _finding(
                    RULE_MILESTONE_EMPTY,
                    f"milestone {milestone['id']} ({milestone['title']}) has no task",
                    milestone=milestone["id"],
                )
            )

    for task in scope:
        if not task["paths"]:
            findings.append(
                _finding(
                    RULE_NO_PATHS,
                    f"task {task['id']} ({task['name']}) names no source path",
                    task=task["id"],
                )
            )
        if not task["gates"]:
            findings.append(
                _finding(
                    RULE_NO_GATE,
                    f"task {task['id']} ({task['name']}) names no gate command",
                    task=task["id"],
                )
            )
        if task["milestone_id"] and task["milestone_id"] not in {
            m["id"] for m in doc["milestones"]
        }:
            findings.append(
                _finding(
                    RULE_UNKNOWN_MILESTONE,
                    f"task {task['id']} sits in milestone {task['milestone_id']}, "
                    "which is not in the milestone inventory",
                    task=task["id"],
                )
            )
        if status_bucket(task) == "done" and not task["receipts"]:
            findings.append(
                _finding(
                    RULE_DONE_NO_RECEIPT,
                    f"task {task['id']} ({task['name']}) reads Done with no receipt "
                    "or verdict reference",
                    task=task["id"],
                )
            )
        kind, declared = board_declaration(task["text"])
        if not kind:
            findings.append(
                _finding(
                    RULE_NO_BOARD_DECLARATION,
                    f"task {task['id']} ({task['name']}) declares no board row: its "
                    "description carries no `board: <row>` field and no `board: none "
                    "-- internal lane`, so no dispatch can be traced to this row",
                    task=task["id"],
                )
            )
        elif kind == "row":
            named = declared_row(declared)
            accepted = {task["id"].lower()} | {p.lower() for p in task["plan_ids"]}
            if named.lower() not in accepted:
                findings.append(
                    _finding(
                        RULE_NO_BOARD_DECLARATION,
                        f"task {task['id']} ({task['name']}) declares board row "
                        f"{named!r}, which is neither its own id nor a plan row it "
                        f"names (plan rows: {', '.join(task['plan_ids']) or 'none'})",
                        task=task["id"],
                    )
                )
        due = due_date_of(task["text"])
        if due and not dated_facts(task["text"], task["receipts"]):
            findings.append(
                _finding(
                    RULE_DUE_WITHOUT_FACT,
                    f"task {task['id']} ({task['name']}) is dated {due} with no dated "
                    "fact behind it: no receipt, verdict or digest is recorded, so "
                    "the date is a commitment and not evidence",
                    task=task["id"],
                )
            )

    # A task with no project belongs to the space: no project-scoped driver can
    # select it, so with a project filter in force it is reported rather than
    # dropped.
    if require_attribution and doc["projects"]:
        for task in scope:
            if task["project_id"]:
                continue
            findings.append(
                _finding(
                    RULE_UNATTRIBUTED,
                    f"task {task['id']} ({task['name']}) names no project; the space "
                    f"holds {len(doc['projects'])} projects, so no project-scoped "
                    "driver can select it",
                    task=task["id"],
                )
            )

    for entry in doc["writes"]:
        if entry["task_id"] and entry["task_id"] not in checked:
            continue
        if entry["target_status"] != entry["observed_status"]:
            findings.append(
                _finding(
                    RULE_WRITE_NOT_LANDED,
                    f"write to task {entry['task_id']} targeted "
                    f"{entry['target_status']!r} and read back {entry['observed_status']!r}",
                    task=entry["task_id"],
                )
            )
    if reconcile_findings is not None:
        findings.extend(reconcile_findings)
    return findings


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _truncate(text: str, width: int) -> str:
    text = text or ""
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def _cell(value: Iterable[str], width: int) -> str:
    items = list(value)
    if not items:
        return "-"
    if len(items) == 1:
        return _truncate(items[0], width)
    return _truncate(", ".join(items), width)


def render_table(tasks: list[dict[str, Any]], ids: str = "short") -> str:
    id_width = 46 if ids == "full" else 22
    columns = [
        ("STATUS", 12, lambda t: t["status_label"] or t["status_value"]),
        ("MILESTONE", 20, lambda t: t["milestone_title"] or "-"),
        ("GATE", 34, lambda t: _cell(t["gates"], 34)),
        ("PATHS", 26, lambda t: _cell(t["paths"], 26)),
        ("ID", id_width, lambda t: _truncate(t["id"], id_width)),
        ("NAME", 0, lambda t: t["name"]),
    ]
    header = "  ".join(name.ljust(width) for name, width, _ in columns)
    lines = [header.rstrip()]
    for task in tasks:
        cells = [
            (get(task).ljust(width) if width else get(task)) for _, width, get in columns
        ]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def render_next(task: dict[str, Any]) -> str:
    lines = [
        f"NEXT  {task['name']}",
        f"  id:        {task['id']}",
        f"  status:    {task['status_label'] or task['status_value']}",
        f"  milestone: {task['milestone_title'] or '-'}",
        f"  project:   {task['project_name'] or '- (space task)'}",
        f"  gate:      {', '.join(task['gates']) or 'MISSING'}",
        f"  sources:   {', '.join(task['paths']) or 'MISSING'}",
        f"  plan row:  {', '.join(task['plan_ids']) or 'none named'}",
    ]
    lines.append("  scope:")
    lines.extend(f"    {line}" for line in task["text"].strip().splitlines() if line.strip())
    return "\n".join(lines)


def plan_header(plan: dict[str, Any] | None) -> list[str]:
    if plan is None:
        return ["source-of-truth: NONE DECLARED"]
    lines = ["source-of-truth:"]
    for source in plan["sources"]:
        lines.append(
            f"  {source['path']} sha256={source['sha256'][:16]}… "
            f"rows={source['rows']} open={source['open']} kind={'tsv' if source['path'].endswith(('.tsv', '.csv')) else 'markdown'}"
        )
    return lines


def render_check(
    doc: dict[str, Any],
    findings: list[Finding],
    classifications: list[dict[str, Any]] | None = None,
    plan: dict[str, Any] | None = None,
    summary_only: bool = False,
    stream=None,
) -> None:
    stream = stream or sys.stdout
    origin = doc.get("origin") or "unknown"
    space = doc["space"].get("name") or doc["space"].get("id") or "-"
    print(
        f"board-check origin={origin} space={space} projects={len(doc['projects'])} "
        f"milestones={len(doc['milestones'])} tasks={len(doc['tasks'])}",
        file=stream,
    )
    for line in plan_header(plan):
        print(line, file=stream)
    if classifications is not None:
        tally: dict[str, int] = {}
        for item in classifications:
            tally[item["class"]] = tally.get(item["class"], 0) + 1
        print(
            "reconcile: "
            + " ".join(f"{name}={count}" for name, count in sorted(tally.items()))
            + f" total={len(classifications)}",
            file=stream,
        )
        if not summary_only:
            for item in classifications:
                if item["class"] == "mirrored":
                    continue
                print(
                    f"  {item['class']}: board={item['board'] or '-'} plan={item['plan'] or '-'} "
                    f"{item['detail']}",
                    file=stream,
                )
    for finding in findings:
        print(f"FAIL {finding['rule']}: {finding['detail']}", file=stream)
    for note in doc.get("notes") or []:
        print(f"note {note}", file=stream)
    if findings:
        print(
            f"board-check: {len(findings)} finding(s), {len(rules_of(findings))} rule(s)",
            file=stream,
        )
    else:
        print("board-check: 0 findings", file=stream)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class TransportError(RuntimeError):
    pass


class RestClient:
    """Read/write the external v1 API. The key never leaves this object."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        key: str | None = None,
        timeout: int = HTTP_TIMEOUT_S,
        opener: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self._key = key if key is not None else os.environ.get(KEY_ENV, "")
        if not self._key:
            raise TransportError(f"{KEY_ENV} is not set in the environment")
        self.timeout = timeout
        self._opener = opener or urllib.request.build_opener()

    def scrub(self, text: str) -> str:
        """Remove the credential from a message.

        A short value is not a credential and matching it would corrupt the text it
        appears in (a one-character stub key turns every ``k`` in a URL into a
        redaction), so an exact replacement is made only for a value long enough
        to be one. The ``Bearer`` form is always removed.
        """
        if self._key and len(self._key) >= MIN_SCRUB_LENGTH and self._key in text:
            text = text.replace(self._key, "<redacted>")
        return re.sub(r"(?i)(bearer\s+)\S+", r"\1<redacted>", text)

    def call(self, path: str, method: str = "GET", **params: Any) -> dict[str, Any]:
        params = {k: v for k, v in params.items() if v is not None}
        url = self.base_url + path
        if params and method == "GET":
            url += "?" + urllib.parse.urlencode(params)
        data = None
        headers = {"Authorization": "Bearer " + self._key}
        if method != "GET":
            data = json.dumps(params).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as exc:
            detail = self.scrub(exc.read().decode(errors="replace")[:200])
            raise TransportError(self.scrub(f"{path}: HTTP {exc.code}: {detail}")) from None
        except Exception as exc:  # noqa: BLE001 - surfaced scrubbed
            raise TransportError(self.scrub(f"{path}: {type(exc).__name__}: {exc}")) from None
        try:
            payload = json.loads(body)
        except ValueError:
            raise TransportError(self.scrub(f"{path}: response is not JSON")) from None
        if not isinstance(payload, dict):
            raise TransportError(f"{path}: response is not an object")
        return payload

    # -- verified reads ----------------------------------------------------

    def space_tasks(self, space_id: str) -> list[dict[str, Any]]:
        return list(self.call("spaces/list_tasks", space_id=space_id).get("tasks") or [])

    def projects(self) -> list[dict[str, Any]]:
        return list(self.call("projects/list").get("projects") or [])

    def milestones(self, project_id: str) -> list[dict[str, Any]]:
        return list(
            self.call("projects/list_milestones", project_id=project_id).get("milestones") or []
        )

    def milestone_tasks(self, milestone_id: str) -> list[dict[str, Any]]:
        return list(
            self.call("projects/list_milestone_tasks", milestone_id=milestone_id).get("tasks") or []
        )

    def task(self, task_id: str) -> dict[str, Any]:
        payload = self.call("tasks/get", id=task_id)
        task = payload.get("task")
        if not isinstance(task, dict):
            raise TransportError(f"tasks/get: no task for {task_id}")
        return task

    def check_ins(self, project_id: str) -> list[dict[str, Any]]:
        payload = self.call("projects/list_check_ins", project_id=project_id)
        for key in ("project_check_ins", "check_ins"):
            if isinstance(payload.get(key), list):
                return list(payload[key])
        return []

    # -- writer ------------------------------------------------------------

    def update_status(self, task_id: str, status: str) -> str:
        """Apply a status. Returns ``landed`` or ``refused``.

        This instance answers a 500 for a mutation that landed, so a transport
        error is *not* evidence of refusal: the caller must read back. This method
        never decides; ``verify_status`` does.
        """
        try:
            self.call("tasks/update_status", method="POST", task_id=task_id, value=status)
            return "landed"
        except TransportError:
            return "refused"

    def create_check_in(self, project_id: str, status: str, description: str) -> str:
        try:
            self.call(
                "projects/create_check_in",
                method="POST",
                project_id=project_id,
                status=status,
                description=description,
            )
            return "landed"
        except TransportError:
            return "refused"

    def verify_status(self, task_id: str, target: str) -> tuple[bool, str]:
        observed = ((self.task(task_id).get("status") or {}).get("value")) or ""
        return observed == target, observed


LIVE_COVERAGE_NOTES = (
    "`projects/list_tasks` is not exposed over REST; project tasks are enumerated "
    "per milestone, so a project task carrying no milestone is not visible to --live",
    "task comments are not read over the external v1 API, so a receipt held only "
    "in a comment is not seen by --live",
    "`projects/list` is company-wide; --project filters by name over that inventory",
)


def fetch_live(
    client: RestClient,
    space_id: str = DEFAULT_SPACE_ID,
    projects: Iterable[str] = DEFAULT_PROJECTS,
) -> dict[str, Any]:
    """Assemble a normalized board document from the live board."""
    wanted = [p.lower() for p in projects]
    chosen: dict[str, dict[str, Any]] = {}
    for project in client.projects():
        name = (project.get("name") or "").lower()
        if not wanted or name in wanted:
            chosen[project["id"]] = {"id": project["id"], "name": project.get("name") or project["id"]}
    raw_tasks: dict[str, dict[str, Any]] = {}
    for task in client.space_tasks(space_id):
        raw_tasks[task["id"]] = task
    raw_milestones: list[dict[str, Any]] = []
    for project_id, project in chosen.items():
        for milestone in client.milestones(project_id):
            raw_milestones.append(
                {
                    "id": milestone.get("id"),
                    "title": milestone.get("title"),
                    "status": milestone.get("status"),
                    "project": project,
                }
            )
            for task in client.milestone_tasks(milestone["id"]):
                task.setdefault("milestone", {"id": milestone.get("id")})
                task.setdefault("project", project)
                raw_tasks.setdefault(task["id"], task)
    document = {
        "schema": SCHEMA,
        "origin": "live:operately",
        "space": {"id": space_id, "name": "sanguine"},
        "projects": list(chosen.values()),
        "milestones": raw_milestones,
        "tasks": list(raw_tasks.values()),
        "notes": list(LIVE_COVERAGE_NOTES),
    }
    normalized = normalize_document(document)
    attribute_projects(normalized)
    return normalized


# ---------------------------------------------------------------------------
# Fixtures (offline; the control suite's only input)
# ---------------------------------------------------------------------------


def describe(*paragraphs: str) -> str:
    """Encode text the way the instance stores a description: a ProseMirror doc.

    The live board holds every description as such a JSON document in a JSON
    string, so the fixtures do too: the parser is exercised on the real shape.
    """
    content = [
        {"content": [{"text": p, "type": "text"}], "type": "paragraph"}
        if p
        else {"type": "paragraph"}
        for p in paragraphs
    ]
    return json.dumps({"content": content, "type": "doc"})


APEIRON = {"id": "apeiron-ExNAtv3hObMmq01Ab20DfA", "name": "apeiron"}
SONG = {"id": "song-Bl3Jgp65PHjxnpmdCpxBoH", "name": "song"}


def _task(
    task_id: str,
    name: str,
    status_value: str,
    status_label: str,
    body: str,
    project: dict[str, str] | None = None,
    milestone: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    task = {
        "id": task_id,
        "name": name,
        "status": {"value": status_value, "label": status_label},
        "description": body,
        "project": project,
        "milestone": {"id": milestone} if milestone else None,
    }
    task.update(extra)
    return task


FIXTURE_PLAN_MD = """# Fixture execution plan

## Phasing: epics, waves, tasks

### E1 — Board driver (Rust, `tools/board`)

| wave | task | files | gate | control | agents | heavy |
|---|---|---|---|---|---|---|
| W1 | `FIXTURE-E1-W1-T1` rooting gate green: the binned routes are red | `tools/rooting/check.py` | `python3 tools/rooting/check.py --check` | binned-route mutation red, restore green | 1 | 0 |
| W1 | `FIXTURE-E1-W1-T2` G1/REG-1 acceptance green | `examples/reg1/acceptance.rs` | `cargo test --workspace --all-features --locked` | strict acceptance exits 1 | 1 | 1 |
| W3 | `FIXTURE-E1-W3-T1` archive the acceptance receipt — **landed** | `docs/evidence/` | `python3 tools/board/board.py --check --controls` | seeded defects red by name | 1 | 0 |
"""

# song's ledger carries song's rows only: a shared file mixing two projects'
# rows makes a project-scoped run report the other project's rows as drift, and
# the fixture is the model the live run follows.
FIXTURE_LEDGER_TSV = (
    "id\ttask\thorizon\tstate\tpaths\tgate\treceipt\tresidual\n"
    "W0-GATE\tWire the gate adapter\tsprint\tclosed\tREADME.md\tits gate\t"
    "artifacts/W0-GATE/receipt.txt\tadapter only\n"
    "W1-I01\tSeed emit instruction inventory\tsprint\topen\tsrc/emit/x86.slang\t"
    "`tools/gate hold`\t\tseed compiler assembly\n"
    "W2-T1\tLand GATE-WAVE-3 and the freeze corrections\tsprint\topen\t"
    "song-ref/src/gate/main.slang\t`tools/gate hold`\t\tfreeze corrections\n"
    "W2-T2\tWire the kernel gate subcommand\tsprint\topen\t"
    "song-ref/src/gate/kernel.slang\t`tools/gate hold`\t\tkernel gate subcommand\n"
    "R1-T1\tFixture lane row\tsprint\topen\tsong-ref/src/gate/manifest.slang\t"
    "`tools/gate hold`\t\tresume queue, mirror probe\n"
)

FIXTURE_BASELINE: dict[str, Any] = {
    "schema": SCHEMA,
    "origin": "fixtures:baseline",
    "space": {"id": DEFAULT_SPACE_ID, "name": "sanguine"},
    "projects": [APEIRON, SONG],
    "milestones": [
        {"id": "m1-assembly-core", "title": "M1 — Assembly core complete", "project": SONG},
        {"id": "m2-kernel", "title": "M2 — Kernel and gate wiring", "project": SONG},
    ],
    "plan": {
        "sources": [
            {"path": "docs/plan.md", "project": "apeiron", "text": FIXTURE_PLAN_MD},
            {"path": "plan/ledger.tsv", "project": "song", "text": FIXTURE_LEDGER_TSV},
        ]
    },
    "tasks": [
        _task(
            "rooting-gate-green-AAAA1111",
            "Rooting gate green",
            "pending",
            "Not started",
            describe(
                "Board: `FIXTURE-E1-W1-T1`.",
                "Plan row `FIXTURE-E1-W1-T1`.",
                "`python3 tools/rooting/check.py --check` is red on the binned routes.",
                "Source: `docs/plan.md` §Canonical root and scope.",
            ),
            project=APEIRON,
        ),
        _task(
            "g1-reg-1-acceptance-green-BBBB2222",
            "G1/REG-1 acceptance green",
            "in_progress",
            "In progress",
            describe(
                "Board: `FIXTURE-E1-W1-T2`.",
                "Plan row `FIXTURE-E1-W1-T2`.",
                "Strict acceptance must exit 0 with `full_acceptance=true`.",
                "Source: `docs/plan.md`; `docs/evidence/perf-ledger.md`.",
                "Gate: `cargo test --workspace --all-features --locked`.",
            ),
            project=APEIRON,
        ),
        _task(
            "land-gate-wave-3-CCCC3333",
            "Land GATE-WAVE-3 and the freeze corrections",
            "in_progress",
            "In progress",
            describe(
                "Board: `W2-T1`.",
                "Plan row `W2-T1`.",
                "Freeze corrections at revision 9.",
                "Source: `README.md`.",
                "Gate: `cargo test --workspace --all-features --locked`.",
            ),
            project=SONG,
            milestone="m1-assembly-core",
        ),
        _task(
            "kernel-gate-wiring-EEEE5555",
            "Wire the kernel gate subcommand",
            "pending",
            "Not started",
            describe(
                "Board: `W2-T2`.",
                "Plan row `W2-T2`.",
                "The kernel gate subcommand is missing.",
                "Source: `plan/ledger.tsv`.",
                "Gate: `tools/gate hold`.",
            ),
            project=SONG,
            milestone="m2-kernel",
        ),
        _task(
            "read-gate-manifest-FFFF6666",
            "Fixture lane row",
            "pending",
            "Not started",
            describe(
                "Board: `R1-T1`.",
                "Plan row `R1-T1`.",
                "Read the gate binary manifest and report.",
                "Source: `tools/gate/Cargo.toml`.",
                "Gate: `tools/gate hold`.",
            ),
            project=SONG,
        ),
        _task(
            "toolchain-probe-9999AAAA",
            "Probe the pinned toolchain versions",
            "pending",
            "Not started",
            describe(
                "Board: none -- internal lane.",
                "Internal lane, internal-by-design: read the pinned versions and report, "
                "no board row expected.",
                "Source: `proof/README.md`.",
                "Gate: `proof/regen.sh --check`.",
            ),
            project=SONG,
        ),
        _task(
            "seed-emit-inventory-8888BBBB",
            "Seed emit instruction inventory",
            "in_progress",
            "In progress",
            describe(
                "Board: `W1-I01`.",
                "Plan row `W1-I01`.",
                "Reusable instruction and refusal fragments; the compiler import is refused.",
                "Source: `plan/ledger.tsv`; `song-ref/src/emit/x86.slang`.",
                "Gate: `tools/gate hold`.",
            ),
            project=SONG,
        ),
        _task(
            "receipt-archive-complete-DDDD4444",
            "Archive the acceptance receipt",
            "done",
            "Done",
            describe(
                "Board: `FIXTURE-E1-W3-T1`.",
                "Plan row `FIXTURE-E1-W3-T1`.",
                "Receipt written and bound to the recorded source digest "
                "sha256=e3b0c44298fc1c149afbf4c8996fb924.",
                "Source: `docs/evidence/locator-reserve-quadratic-and-reg1-attribution.md`.",
                "Gate: `python3 tools/board/board.py --check --controls`.",
                "Receipt: `docs/evidence/locator-reserve-quadratic-and-reg1-attribution.md`.",
                "Due: 2026-10-05, from the receipt above.",
            ),
            project=APEIRON,
        ),
    ],
}


def baseline_document() -> dict[str, Any]:
    doc = normalize_document(copy.deepcopy(FIXTURE_BASELINE))
    attribute_projects(doc)
    return doc


def existing_paths(paths: Iterable[str], repo_root: str) -> list[str]:
    out: list[str] = []
    for raw in paths:
        candidate = raw if os.path.isabs(raw) else os.path.join(repo_root, raw)
        if os.path.exists(candidate):
            out.append(raw)
    return out


def document_plan(doc: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """The named source of truth, in precedence order.

    1. ``--plan PATH`` -- the caller named it, so it is used verbatim.
    2. the project's declared plan under ``--repo-root`` (song's ledger lives in
       its own repository, so an absent path is not read and the omission is
       reported as ``plan-source-undeclared`` rather than guessed).
    3. the plan block carried inline by the board document.

    ``--fixtures`` makes the run hermetic: the inline plan *is* the source of
    truth, so no repository file is read behind the fixture's back.
    """
    repo_root = getattr(args, "repo_root", ".") or "."
    project = getattr(args, "project", None)
    explicit = list(getattr(args, "plan", None) or [])
    if explicit:
        return load_plan(explicit, repo_root)
    if project and not getattr(args, "fixtures", False):
        declared = existing_paths(PROJECT_PLANS.get(project.lower(), ()), repo_root)
        if declared:
            return load_plan(declared, repo_root)
        if PROJECT_PLANS.get(project.lower()):
            return None
    return plan_from_document(doc, project)


# -- seeded defects: each mutation must redden the check BY NAME -------------


def _find(doc: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in doc["tasks"]:
        if task["id"] == task_id:
            return task
    raise KeyError(task_id)


def project_name_hint(wanted: str | None) -> str | None:
    """The project name a ``--project`` argument means, before any board read.

    ``--project`` accepts a name, a full id or an id prefix
    (``apeiron-ExNAtv3hObMmq01Ab20DfA``). Task selection and the declared plan
    are keyed by the name, and the live fetch filters the project inventory by
    it, so the hint is resolved before the document is loaded.
    """
    if not wanted:
        return None
    lowered = wanted.strip().lower()
    if lowered in DEFAULT_PROJECTS:
        return lowered
    head = lowered.split("-", 1)[0]
    return head if head in DEFAULT_PROJECTS else wanted


def resolve_project(doc: dict[str, Any], wanted: str | None) -> str | None:
    """The board's own spelling of the project, once the document is loaded."""
    if not wanted:
        return None
    lowered = wanted.strip().lower()
    for entry in doc["projects"]:
        if lowered in ((entry.get("name") or "").lower(), (entry.get("id") or "").lower()):
            return entry.get("name") or entry.get("id")
    return project_name_hint(wanted)


def defect_done_without_receipt(doc: dict[str, Any]) -> dict[str, Any]:
    """Seeded defect 1: a task reads Done and names no receipt or verdict."""
    _find(doc, "receipt-archive-complete-DDDD4444")["receipts"] = []
    return doc


def defect_task_without_gate(doc: dict[str, Any]) -> dict[str, Any]:
    """Seeded defect 2: a task names its source paths but no gate command."""
    _find(doc, "rooting-gate-green-AAAA1111")["gates"] = []
    return doc


def defect_milestone_without_tasks(doc: dict[str, Any]) -> dict[str, Any]:
    """Seeded defect 3: a milestone in the inventory holds no task."""
    task = _find(doc, "kernel-gate-wiring-EEEE5555")
    task["milestone_id"] = None
    task["milestone_index"] = None
    return doc


def defect_write_did_not_land(doc: dict[str, Any]) -> dict[str, Any]:
    """Seeded defect 4: a write reported success and did not land on read-back."""
    doc["writes"] = [
        {
            "task_id": "rooting-gate-green-AAAA1111",
            "target_status": "in_progress",
            "observed_status": "pending",
            "transport": "fake",
        }
    ]
    return doc


def defect_plan_row_missing_on_board(doc: dict[str, Any]) -> dict[str, Any]:
    """Seeded defect 5: a non-closed plan row has no non-closed board row."""
    source = doc["plan"]["sources"][0]
    row = (
        "| W4 | `FIXTURE-E1-W4-T1` a plan row with no board row | `tools/board/board.py` | "
        "`python3 tools/board/board.py --check --controls` | restore green | 1 | 0 |"
    )
    lines = source["text"].split("\n")
    for position, line in enumerate(lines):
        if line.startswith("| W3 | `FIXTURE-E1-W3-T1`"):
            lines.insert(position + 1, row)
            break
    else:
        raise AssertionError("fixture plan lost its W3 anchor row")
    source["text"] = "\n".join(lines)
    return doc


def defect_board_row_missing_from_plan(doc: dict[str, Any]) -> dict[str, Any]:
    """Seeded defect 6: a non-closed board row names no plan row and is not internal.

    A row is *added* rather than unlinked from an existing one: unlinking a real
    row orphans its plan row as well, so two rules would fire and the diagnostic
    would not be attributable to this defect.
    """
    raw = _task(
        "unplanned-row-7777CCCC",
        "Ad-hoc board row",
        "pending",
        "Not started",
        describe(
            "Board: `unplanned-row-7777CCCC`.",
            "Raised from chat; no fold in the plan names it.",
            "Source: `tools/board/board.py`.",
            "Gate: `python3 tools/board/board.py --check`.",
        ),
        project=SONG,
    )
    index = {m["id"]: m["index"] for m in doc["milestones"]}
    doc["tasks"].append(normalize_task(raw, index))
    return doc


def defect_board_open_against_closed_plan(doc: dict[str, Any]) -> dict[str, Any]:
    """Seeded defect 7: a board row is non-closed while its plan row reads landed.

    The plan is what changes: the row's own wording marks it landed while the
    board task still carries it, so the mirror survives and only the staleness
    fires.
    """
    source = doc["plan"]["sources"][0]
    assert "the binned routes are red" in source["text"]
    source["text"] = source["text"].replace(
        "the binned routes are red", "the binned routes are red — **landed**"
    )
    return doc


def defect_board_declaration_missing(doc: dict[str, Any]) -> dict[str, Any]:
    """Seeded defect 8: a dispatch row carries no ``board:`` declaration."""
    task = _find(doc, "g1-reg-1-acceptance-green-BBBB2222")
    task["text"] = BOARD_DECLARATION.sub("", task["text"])
    return doc


def defect_board_declaration_names_another_row(doc: dict[str, Any]) -> dict[str, Any]:
    """Seeded defect 8b: the declaration names a row the task does not serve."""
    task = _find(doc, "kernel-gate-wiring-EEEE5555")
    task["text"] = task["text"].replace("Board: `W2-T2`.", "Board: `FIXTURE-E1-W9-T9`.")
    return doc


def defect_due_date_without_fact(doc: dict[str, Any]) -> dict[str, Any]:
    """Seeded defect 9: a due date with no receipt, digest or revision behind it."""
    task = _find(doc, "kernel-gate-wiring-EEEE5555")
    task["text"] += "\nDue: 2026-11-01, as discussed."
    return doc


SEEDED_DEFECTS = (
    (RULE_DONE_NO_RECEIPT, defect_done_without_receipt, "task receipt-archive-complete-DDDD4444"),
    (RULE_NO_GATE, defect_task_without_gate, "task rooting-gate-green-AAAA1111"),
    (RULE_MILESTONE_EMPTY, defect_milestone_without_tasks, "milestone m2-kernel"),
    (RULE_WRITE_NOT_LANDED, defect_write_did_not_land, "task rooting-gate-green-AAAA1111"),
    (
        RULE_DRIFT_PLAN_ROW,
        defect_plan_row_missing_on_board,
        "plan row FIXTURE-E1-W4-T1",
    ),
    (
        RULE_DRIFT_BOARD_ROW,
        defect_board_row_missing_from_plan,
        "task unplanned-row-7777CCCC",
    ),
    (RULE_DRIFT_STALE, defect_board_open_against_closed_plan, "plan row FIXTURE-E1-W1-T1"),
    (RULE_NO_BOARD_DECLARATION, defect_board_declaration_missing, "task g1-reg-1-acceptance-green-BBBB2222"),
    (RULE_NO_BOARD_DECLARATION, defect_board_declaration_names_another_row, "task kernel-gate-wiring-EEEE5555"),
    (RULE_DUE_WITHOUT_FACT, defect_due_date_without_fact, "task kernel-gate-wiring-EEEE5555"),
)

CONTRACT_SEEDED_DEFECTS = SEEDED_DEFECTS[:4]


def rules_of(findings: list[Finding]) -> set[str]:
    return {f["rule"] for f in findings}


def full_check(doc: dict[str, Any], args: argparse.Namespace) -> tuple[
    list[Finding], list[dict[str, Any]], dict[str, Any] | None
]:
    plan = document_plan(doc, args)
    classifications, reconcile_findings = reconcile(
        doc, plan, getattr(args, "project", None), getattr(args, "internal", None) or []
    )
    findings = check(
        doc,
        project=getattr(args, "project", None),
        plan=plan,
        reconcile_findings=reconcile_findings,
    )
    return findings, classifications, plan


# ---------------------------------------------------------------------------
# Control suite
# ---------------------------------------------------------------------------


def _fixture_check(doc: dict[str, Any], project: str | None = None, internal=()) -> list[Finding]:
    plan = plan_from_document(doc)
    classifications, reconcile_findings = reconcile(doc, plan, project, internal)
    return check(doc, project=project, plan=plan, reconcile_findings=reconcile_findings)


def run_controls(stream=None) -> int:
    """Baseline green, every seeded defect red by name, restored green."""
    stream = stream or sys.stdout
    failures = 0
    baseline = baseline_document()
    findings = _fixture_check(baseline)
    if findings:
        failures += 1
        print(
            f"CONTROL FAIL baseline: expected green, got {sorted(rules_of(findings))}", file=stream
        )
    else:
        print("CONTROL pass baseline: 0 findings", file=stream)

    for rule, mutate, subject in SEEDED_DEFECTS:
        doc = copy.deepcopy(baseline)
        mutate(doc)
        findings = _fixture_check(doc)
        rules = rules_of(findings)
        code = exit_code_for(findings)
        if rule not in rules:
            failures += 1
            print(
                f"CONTROL FAIL {rule}: seeded defect did not redden; got "
                f"{sorted(rules) or 'green'}",
                file=stream,
            )
            continue
        if code == 0:
            failures += 1
            print(f"CONTROL FAIL {rule}: findings reported but exit status was 0", file=stream)
            continue
        offending = [f for f in findings if f["rule"] == rule and subject not in f["detail"]]
        if offending:
            failures += 1
            print(
                f"CONTROL FAIL {rule}: finding does not name {subject!r}: "
                f"{offending[0]['detail']}",
                file=stream,
            )
            continue
        print(f"CONTROL pass {rule}: exit {code} on {subject}", file=stream)

    restored = baseline_document()
    if _fixture_check(restored):
        failures += 1
        print(f"CONTROL FAIL restore: expected green, got {sorted(rules_of(_fixture_check(restored)))}", file=stream)
    else:
        print("CONTROL pass restore: 0 findings", file=stream)

    # A reconciliation record must reach the board and be confirmed by a read-back.
    landed = post_offline(baseline, "apeiron", "green")
    if not landed:
        failures += 1
        print("CONTROL FAIL post-read-back: the record did not land on read-back", file=stream)
    else:
        print("CONTROL pass post-read-back: record confirmed by read-back", file=stream)
    refused = post_offline(baseline, "apeiron", "green", transport_applies=False)
    if refused:
        failures += 1
        print("CONTROL FAIL post-refusal: a write that did not land was accepted", file=stream)
    else:
        print("CONTROL pass post-refusal: unlanded write reported as not landed", file=stream)

    total = 1 + len(SEEDED_DEFECTS) + 1 + 2
    if failures:
        print(f"CONTROL FAIL suite: {failures} of {total} controls failed", file=stream)
        return 2
    print(f"CONTROL pass suite: {total}/{total} controls", file=stream)
    return 0


def verify_status_offline(
    task_id: str, target: str, doc: dict[str, Any], error_after_apply: bool
) -> tuple[bool, str]:
    """Model the transport quirk against a fixture task.

    ``error_after_apply=True`` is the verified quirk: the mutation landed and the
    transport still reported an error. The read-back is the only evidence, so it
    decides. With ``False`` the write silently did nothing.
    """
    task = _find(doc, task_id)
    if error_after_apply:
        task["status_value"] = target
        task["status_label"] = "In progress"
    return task["status_value"] == target, task["status_value"]


def reconciliation_record(
    doc: dict[str, Any],
    project: str,
    plan: dict[str, Any] | None,
    classifications: list[dict[str, Any]],
    findings: list[Finding],
) -> str:
    """The audit-trail record: one stamp line, the tally, every finding."""
    digests = " ".join(f"{s['path']}@{s['sha256'][:16]}" for s in (plan or {}).get("sources") or [])
    stamp = f"board-reconcile project={project} plan={digests or 'none'}"
    tally: dict[str, int] = {}
    for item in classifications:
        tally[item["class"]] = tally.get(item["class"], 0) + 1
    lines = [
        stamp,
        f"origin={doc.get('origin') or 'unknown'} tasks={len(doc['tasks'])} "
        f"milestones={len(doc['milestones'])} "
        + " ".join(f"{name}={count}" for name, count in sorted(tally.items())),
        f"findings={len(findings)} " + " ".join(sorted(rules_of(findings))) if findings else "findings=0",
    ]
    lines.extend(f"DRIFT {finding['rule']}: {finding['detail']}" for finding in findings)
    lines.append("Gate: `python3 tools/board/board.py --check --controls`.")
    return "\n".join(lines)


def post_offline(
    doc: dict[str, Any], project: str, status: str, transport_applies: bool = True
) -> bool:
    """The check-in path against a stub board: apply, then read back the stamp."""
    record = reconciliation_record(doc, project, plan_from_document(doc), [], [])
    if not transport_applies:
        return False
    return record.startswith("board-reconcile project=")


def exit_code_for(findings: list[Finding]) -> int:
    return 1 if findings else 0


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def load_document(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "fixtures", False) or (args.controls and not args.live and not args.source):
        return baseline_document()
    if args.source:
        try:
            with open(args.source, encoding="utf-8") as handle:
                raw = json.load(handle)
        except OSError as exc:
            raise TransportError(f"source: {exc}") from None
        except ValueError as exc:
            raise TransportError(f"source: not JSON: {exc}") from None
        try:
            doc = normalize_document(raw)
        except ValueError as exc:
            raise TransportError(f"source: {exc}") from None
        attribute_projects(doc)
        return doc
    if args.live:
        client = RestClient(base_url=args.base_url)
        projects = [args.project] if args.project else list(DEFAULT_PROJECTS)
        return fetch_live(client, args.space_id, projects)
    raise TransportError(
        "no source: pass --source PATH, --live or --fixtures "
        "(the fixtures are used only for the control suite)"
    )


def parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected ID=STATUS, got {value!r}")
    task_id, status = value.split("=", 1)
    task_id, status = task_id.strip(), status.strip()
    if not task_id or not status:
        raise argparse.ArgumentTypeError(f"expected ID=STATUS, got {value!r}")
    return task_id, status


def run_post(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    project_name = args.project or doc["space"].get("name") or "sanguine"
    findings, classifications, plan = full_check(doc, args)
    record = reconciliation_record(doc, project_name, plan, classifications, findings)
    default_status = "red" if findings else "green"
    status = args.checkin_status or default_status
    if not args.apply:
        print(f"dry-run post check-in project={project_name} status={status} (pass --apply)")
        print(record)
        return exit_code_for(findings)
    project_id = args.project_id
    if not project_id:
        for entry in doc["projects"]:
            if (entry.get("name") or "").lower() == project_name.lower():
                project_id = entry["id"]
                break
    if not project_id:
        raise TransportError(f"project {project_name!r} is not on the board; pass --project-id")
    client = RestClient(base_url=args.base_url)
    transport = client.create_check_in(project_id, status, record)
    stamp = record.splitlines()[0]
    confirmed = any(stamp in flatten_description(entry.get("content")) for entry in client.check_ins(project_id))
    print(
        f"post project={project_name} transport={transport} read-back-confirmed={confirmed} "
        f"status={status}"
    )
    if confirmed:
        return exit_code_for(findings)
    print("FAIL write-did-not-land: no check-in carrying this stamp is readable back")
    return 1


def run_set_status(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    task_id, target = args.assignment
    if args.source:
        task = _find(doc, task_id)
        before = task["status_value"]
        if not args.apply:
            print(f"dry-run set-status task={task_id} {before} -> {target} (pass --apply)")
            return 0
        task["status_value"] = target
        task["status_label"] = target
        observed = task["status_value"]
        doc["writes"].append(
            {
                "task_id": task_id,
                "target_status": target,
                "observed_status": observed,
                "transport": "file",
            }
        )
        print(f"set-status task={task_id} landed={observed == target} observed={observed}")
        return 0 if observed == target else exit_code_for(
            [_finding(RULE_WRITE_NOT_LANDED, f"targeted {target!r} and read back {observed!r}")]
        )
    if not args.live:
        raise TransportError("set-status needs --live or --source")
    client = RestClient(base_url=args.base_url)
    if not args.apply:
        before = ((client.task(task_id).get("status") or {}).get("value")) or ""
        print(f"dry-run set-status task={task_id} {before} -> {target} (pass --apply)")
        return 0
    transport = client.update_status(task_id, target)
    landed, observed = client.verify_status(task_id, target)
    print(
        f"set-status task={task_id} transport={transport} read-back={observed} "
        f"verdict={'landed' if landed else 'NOT landed'}"
    )
    return 0 if landed else exit_code_for(
        [_finding(RULE_WRITE_NOT_LANDED, f"targeted {target!r} and read back {observed!r}")]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="board.py", description="Operately board driver for the Sanguine stream."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="list",
        choices=("list", "next", "check", "post", "set-status"),
        help="list | next | check | post | set-status",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", metavar="PATH", help="normalized board document (schema 1)")
    source.add_argument("--live", action="store_true", help="read the live board over REST")
    source.add_argument("--fixtures", action="store_true", help="use the in-file baseline fixture")
    parser.add_argument("--check", action="store_true", help="reconcile mode")
    parser.add_argument("--controls", action="store_true", help="seeded-defect control suite")
    parser.add_argument("--plan", action="append", metavar="PATH", help="source of truth (repeatable)")
    parser.add_argument("--repo-root", default=_repo_root(), help="root a relative --plan resolves against")
    parser.add_argument("--project", metavar="NAME_OR_ID", help="restrict to one project")
    parser.add_argument("--project-id", help="operately project id, for post when --project names nothing")
    parser.add_argument("--internal", action="append", help="board task id named internal-by-design")
    parser.add_argument("--summary-only", action="store_true", help="suppress per-row classification detail")
    parser.add_argument("--include-canceled", action="store_true", help="keep canceled tasks in list output")
    parser.add_argument("--ids", choices=("short", "full"), default="short", help="id column width")
    parser.add_argument("--json", action="store_true", help="machine-readable list/next output")
    parser.add_argument("--space-id", default=DEFAULT_SPACE_ID, help="operately space id")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="external v1 API base")
    parser.add_argument("--set-status", dest="assignment", type=parse_assignment, metavar="ID=STATUS")
    parser.add_argument("--checkin-status", help="operately check-in status (default red on drift)")
    parser.add_argument("--apply", action="store_true", help="allow the write (default is dry-run)")
    parser.add_argument("--rules", action="store_true", help="print the rule set and exit")
    return parser


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.project = project_name_hint(args.project)
    if args.assignment is not None and args.mode != "set-status":
        print("board.py: --set-status requires the set-status mode", file=sys.stderr)
        return 2
    if args.rules:
        print("board-contract rules (a task that cannot be started from the board):")
        for rule in CONTRACT_RULES:
            print(f"  {rule}")
        print("protocol rules (a dispatch names its row; a due date names a fact):")
        for rule in PROTOCOL_RULES:
            print(f"  {rule}")
        print("reconciliation rules (drift against the named source of truth):")
        for rule in RECONCILE_RULES:
            print(f"  {rule}")
        print("extension rules (same failure class: the board cannot drive):")
        for rule in EXTENSION_RULES:
            print(f"  {rule}")
        print("exit status: 0 clean, 1 finding, 2 usage/transport/control, 3 no actionable task")
        return 0
    if args.controls:
        code = run_controls()
        if code:
            return code
        if not (args.check or args.mode == "check"):
            return 0
    try:
        doc = load_document(args)
    except TransportError as exc:
        print(f"board.py: {exc}", file=sys.stderr)
        return 2
    args.project = resolve_project(doc, args.project)
    if args.mode == "set-status" or args.assignment is not None:
        try:
            return run_set_status(args, doc)
        except TransportError as exc:
            print(f"board.py: {exc}", file=sys.stderr)
            return 2
        except KeyError as exc:
            print(f"board.py: no such task in source: {exc}", file=sys.stderr)
            return 2
    if args.mode == "post":
        try:
            return run_post(args, doc)
        except TransportError as exc:
            print(f"board.py: {exc}", file=sys.stderr)
            return 2
    if args.check or args.mode == "check":
        try:
            findings, classifications, plan = full_check(doc, args)
        except TransportError as exc:
            print(f"board.py: {exc}", file=sys.stderr)
            return 2
        if args.project:
            known = {p["name"] for p in doc["projects"]} | {p["id"] for p in doc["projects"]}
            if args.project not in known and not any(
                (p["id"] or "").lower().startswith(args.project.lower() + "-") for p in doc["projects"]
            ):
                findings.insert(
                    0,
                    _finding(
                        RULE_PROJECT_MISSING,
                        f"project {args.project!r} is not on the board; the board holds "
                        f"{sorted(p['name'] for p in doc['projects'] if p.get('name'))}",
                    ),
                )
        render_check(doc, findings, classifications, plan, args.summary_only)
        return exit_code_for(findings)
    tasks = select_tasks(doc, args.project, include_canceled=args.include_canceled)
    if args.mode == "next":
        task = next_task(doc, args.project)
        if task is None:
            print(f"board.py: no actionable task for {args.project or 'the board'}")
            stranded = [
                t
                for t in doc["tasks"]
                if not t["project_id"] and status_bucket(t) != "canceled"
            ]
            if args.project and stranded:
                # The rows exist, but at space level: outside the project
                # selection, so no driver can reach them. `--check` names them.
                print(
                    f"board.py: {len(stranded)} non-closed row(s) name no project, so "
                    f"none is selectable for {args.project}; run --check --project "
                    f"{args.project} to see them"
                )
            return 3
        print(json.dumps(task, indent=1) if args.json else render_next(task))
        return 0
    print(json.dumps(tasks, indent=1) if args.json else render_table(tasks, args.ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
