#!/usr/bin/env python3
"""board_core: the board contract and the description parser.

`tools/board/board.py` is the entry point every caller runs
(``python3 tools/board/board.py --check --controls``). It dispatches; the
implementation lives in these sibling modules, imported by name, so the command
surface, the findings, the rendered output and the exit codes are unchanged:

  * ``board_core``      schema constants, the rule names, the transport error,
                        description parsing and the finding type.
  * ``board_model``     the named sources of truth, the normalized board
                        document and task selection.
  * ``board_rest``      the external v1 REST client and the live assembly.
  * ``board_fixtures``  the offline baseline the control suite reads.
  * ``board_check``     reconciliation, the contract check, rendering, the seeded
                        defects and the audit-trail record.
  * ``board_controls``  ``--controls``: the seeded-defect suite.

Rule names are the stable interface (``AGENTS.md`` §9.3): the controls assert on
them by name, so a rename breaks a control rather than weakening it.
"""

from __future__ import annotations

import json
import re
from typing import Any
from typing import Iterable

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


class Finding(dict):
    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self['rule']}: {self['detail']}"


def _finding(rule: str, detail: str, **extra: Any) -> Finding:
    item = Finding(rule=rule, detail=detail)
    item.update(extra)
    return item


class TransportError(RuntimeError):
    pass


def rules_of(findings: list[Finding]) -> set[str]:
    return {f["rule"] for f in findings}
