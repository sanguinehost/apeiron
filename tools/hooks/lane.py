#!/usr/bin/env python3
"""The per-commit board-declaration requirement (AGENTS.md §9.3).

A dispatch names its row. Every commit carries the declaration, either in the
message or in the staged change:

    board: <row-id>
    board: none -- housekeeping
    board: none -- spike APEIRON-<EPIC>-SPIKE-<slug>

The field is parsed by ``tools/board/board.py`` itself -- ``board_declaration``,
``declared_row`` and ``slug`` -- so the hook enforces the driver's rule rather
than a second spelling of it. A named row must exist on the board and sit in a
milestone of a project: a row outside a milestone cannot be selected, ordered or
driven, which is the ``unattributed-task`` failure class the driver already
reports.

The declaration is load-bearing: it is matched against the *staged change*, so a
no-row declaration only works where there is no code to attribute. A staged path
under ``CODE_PREFIXES``/``CODE_FILES`` (below) requires ``board: <row-id>``
naming a row that resolves, or a well-formed spike id; ``none -- housekeeping``
covers a change confined to documentation and tooling, and the old phrase
``none -- internal lane`` is retired outright -- refused by name on every
commit, because it named no marker, bounded nothing, and was accepted on 12
consecutive commits of which 8 touched code (``measured``, ``git log -14`` at
``e6f2c90``).

A hook cannot call MCP and no REST move-task route exists, so the hook resolves
and *reports* the row; moving it stays the lead's action at the board.

Two tiers, because git will not give a ``pre-commit`` hook the message: at that
point ``COMMIT_EDITMSG`` still holds the *previous* commit's message and the new
one is written after the hook returns. ``pre-commit`` (``lane.py staged``)
therefore checks what the staged change carries and never demands a field it
cannot see; ``commit-msg`` (``lane.py message PATH``), which git hands the real
message file, is where presence is enforced.

The full reconciliation (``board.py check``) is a wave-boundary gate and is not
run here: it compares the whole board against ``docs/plan.md``, which is red by
design between wave boundaries. The *declaration* is per-commit; the reconcile is
per-wave.

Board source, in precedence order:

1. ``APEIRON_HOOKS_BOARD_SOURCE`` -- a schema-1 board document (offline; see
   ``lane.py capture``). Read the same way ``board.py --source`` reads it.
2. the live board over REST, when ``OPERATELY_API_KEY`` is in the environment.
3. nothing: a commit that names a row cannot be checked, so it is refused by
   ``board-unavailable`` and the fix is printed. ``housekeeping`` and a
   well-formed spike need no board: neither names a row to move, and the
   no-row decision is taken from the staged paths alone, before any board read,
   so a board outage can neither grant nor withhold it.

Modes: ``staged`` (the pre-commit half), ``message PATH`` (the commit-msg half),
``capture`` (write a snapshot), ``rows`` (diagnostic), ``self-test`` (the
controls).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
BOARD_DIR = REPO_ROOT / "tools" / "board"
sys.path.insert(0, str(BOARD_DIR))

try:
    import board  # noqa: E402 - the driver is the authority for the rule and the parse
except ImportError as exc:  # the rule is the driver's; without it nothing is checked
    board = None  # type: ignore[assignment]
    BOARD_IMPORT_ERROR = str(exc)
else:
    BOARD_IMPORT_ERROR = ""

RULE_MISSING = "board-declaration-missing"
RULE_CONFLICT = "board-declaration-conflict"
RULE_NOT_FOUND = "board-row-not-found"
RULE_AMBIGUOUS = "board-row-ambiguous"
RULE_NO_MILESTONE = "board-row-not-in-milestone-bearing-project"
RULE_UNAVAILABLE = "board-unavailable"
RULE_GIT = "staged-state-unreadable"
RULE_DRIVER_MISSING = "board-driver-missing"
RULE_NO_ROW_RETIRED = "board-no-row-retired"
RULE_SPIKE_ID = "board-spike-id-malformed"
RULE_NO_ROW_ON_CODE = "board-no-row-lane-on-code-change"

# A staged path under one of these is code: it compiles, extracts, or is a crate
# manifest, so the change is attributable work and needs a board row. The
# dispatched set was ``src/ tests/ examples/ proof/``; this repository also
# tracks Rust under ``derive/``, ``fold-abi/`` and ``fuzz/`` (``git ls-files``,
# 2026-10-04), and a manifest edit changes what compiles, so all of them are in
# and the refusal is the conservative direction. ``benches/`` is empty in the
# index and is carried for a future Cargo target.
CODE_PREFIXES = (
    "src/",
    "tests/",
    "examples/",
    "proof/",
    "derive/",
    "fold-abi/",
    "fuzz/",
    "benches/",
)
CODE_FILES = ("Cargo.toml", "Cargo.lock")

# The documentation/tooling lane, stated for the diagnostic. It is the
# complement that matters: ``housekeeping`` is legal exactly when the staged set
# is disjoint from CODE_PREFIXES/CODE_FILES, so the rule below does not depend on
# enumerating this set. Anything outside both (README.md, artifacts/, fuzz
# corpora payloads) is also housekeeping-eligible; the residual population is
# named in the refusal text so a commit cannot pass by being unclassified.
DOC_PREFIXES = ("docs/", "tools/", ".githooks/", ".github/")
DOC_FILES = ("AGENTS.md", "README.md", "LICENSE", "VOICE.md")

# The only two no-row declarations. ``board_declaration`` hands back the whole
# remainder of the field, so the marker is matched here against its exact form:
# a trailing reason (``none -- housekeeping (only docs)``) is not the marker.
HOUSEKEEPING = re.compile(r"none\s+--\s*housekeeping\Z", re.I)
SPIKE_DECLARATION = re.compile(r"none\s+--\s*spike\s+(\S+)\Z", re.I)
# AGENTS.md §3: ``APEIRON-<EPIC>-SPIKE-<slug>``, a time limit and a question.
SPIKE_ID = re.compile(r"APEIRON-[A-Z0-9]+(?:-[A-Z0-9]+)*-SPIKE-[a-z0-9]+(?:-[a-z0-9]+)*\Z")

DECLARATION_HINT = (
    "the commit must carry a labelled field on its own line, in the message or "
    "in the staged change:\n"
    "    board: <row-id>\n"
    "    board: none -- housekeeping            (documentation/tooling only)\n"
    "    board: none -- spike APEIRON-<EPIC>-SPIKE-<slug>"
)


class Finding(dict):
    """A named refusal. ``__str__`` is the diagnostic the hook prints."""

    def __str__(self) -> str:
        return f"{self['rule']}: {self['detail']}"


def _finding(rule: str, detail: str, **extra: Any) -> Finding:
    item = Finding(rule=rule, detail=detail)
    item.update(extra)
    return item


# ---------------------------------------------------------------------------
# Git: the staged change and the message being written
# ---------------------------------------------------------------------------


def git(*args: str, root: Path | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(root or REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def git_dir(root: Path | None = None) -> Path:
    return Path(git("rev-parse", "--absolute-git-dir", root=root).strip())


def staged_added_lines(root: Path | None = None) -> list[str]:
    """Added lines of the staged diff, ``+++`` headers excluded."""
    diff = git("diff", "--cached", "-U0", "--no-color", root=root)
    return [
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def staged_paths(root: Path | None = None) -> list[str]:
    """Repository-relative paths in the staged change.

    ``--diff-filter=ACDMRT`` keeps every status, deletions included: moving
    ``src/x.rs`` to ``docs/y.md`` is either one rename (destination only) or a
    delete plus an add, and dropping ``D`` would drop the code half of a move and
    let it declare ``housekeeping``. ``--name-only`` prints the destination path
    alone for a rename; it never emits a brace form.
    """
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACDMRT", root=root)
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_code_path(path: str) -> bool:
    """True when a staged path is code (see CODE_PREFIXES/CODE_FILES)."""
    return path.startswith(CODE_PREFIXES) or path in CODE_FILES


def code_paths(paths: Iterable[str]) -> list[str]:
    """The staged paths that are code, sorted; empty means the commit is docs."""
    return sorted(path for path in paths if is_code_path(path))


def find_declaration(
    texts: Iterable[tuple[str, str]],
) -> tuple[list[tuple[str, str, str]], list[Finding]]:
    """Every ``board:`` field in the named texts.

    Returns ``(declarations, findings)`` where a declaration is
    ``(source, kind, value)``. Absence is the caller's decision, not a finding
    here: ``pre-commit`` may not demand a field it cannot see. A field naming a
    different row in two sources is ``board-declaration-conflict``: a commit
    serves one row.
    """
    declarations: list[tuple[str, str, str]] = []
    for source, text in texts:
        kind, value = board.board_declaration(text or "")
        if kind:
            declarations.append((source, kind, value))
    findings: list[Finding] = []
    rows = {board.declared_row(value).lower() for _, kind, value in declarations if kind == "row"}
    if len(rows) > 1:
        named = ", ".join(sorted(rows))
        sources = ", ".join(source for source, _, _ in declarations)
        findings.append(
            _finding(
                RULE_CONFLICT,
                f"the commit names {len(rows)} different board rows ({named}) across "
                f"{sources}; a commit serves one row",
            )
        )
    return declarations, findings


# ---------------------------------------------------------------------------
# Board resolution
# ---------------------------------------------------------------------------


def load_board_documents(source: str | None, projects: Iterable[str]) -> dict[str, Any]:
    """The board document the check reads: a snapshot, else the live board.

    Raises ``board.TransportError`` when neither is readable.
    """
    if source:
        return board.load_document(_source_namespace(source))
    return board.fetch_live(board.RestClient(), board.DEFAULT_SPACE_ID, list(projects))


def _source_namespace(source: str) -> argparse.Namespace:
    """The driver's own parsed arguments for a snapshot read.

    Built from ``board.build_parser`` so every attribute ``load_document``
    reads is present; a hand-built namespace would drift from the driver.
    """
    return board.build_parser().parse_args(["list", "--source", source])


def match_rows(doc: dict[str, Any], token: str) -> list[tuple[dict[str, Any], str]]:
    """Rows the declared token names, each with the rule that matched it.

    The rules are the driver's own: equality with a row id, a slug match against
    a row name, the ``board:`` field inside the row's own description, or a plan
    id the description carries. A fuzzy match needs a token at least as long as
    ``board.MIN_SLUG_MATCH``, the driver's own floor for a name match.
    """
    low = token.strip().lower()
    strong = len(token.strip()) >= board.MIN_SLUG_MATCH
    token_slug = board.slug(token)
    hits: list[tuple[dict[str, Any], str]] = []
    for row in doc["tasks"]:
        rule = ""
        if (row["id"] or "").lower() == low:
            rule = "row id"
        elif strong and (row["id"] or "").lower().startswith(low + "-"):
            rule = "row id prefix"
        elif strong and token_slug and (
            board.slug(row["name"]) == token_slug
            or board.slug(row["name"]).startswith(token_slug + "-")
        ):
            rule = "row name slug"
        elif any(plan_id.lower() == low for plan_id in row["plan_ids"]):
            rule = "plan id in the row description"
        else:
            kind, value = board.board_declaration(row["text"])
            if kind == "row" and board.declared_row(value).lower() == low:
                rule = "board field in the row description"
        if rule:
            hits.append((row, rule))
    return hits


def milestone_of(doc: dict[str, Any], row: dict[str, Any]) -> tuple[str, str]:
    """``(milestone title, project name)`` for a row, from the board's own records."""
    milestones = {m["id"]: m for m in doc["milestones"]}
    milestone = milestones.get(row["milestone_id"] or "", {})
    title = milestone.get("title") or row["milestone_title"] or ""
    project = milestone.get("project_name") or row["project_name"] or ""
    return title, project


def check_row(doc: dict[str, Any], token: str) -> tuple[list[Finding], list[str]]:
    """Findings and informational notes for a declared row."""
    hits = match_rows(doc, token)
    notes: list[str] = []
    if not hits:
        return (
            [
                _finding(
                    RULE_NOT_FOUND,
                    f"board row {token!r} is not on the board: no row matches it by id, "
                    f"name slug, plan id or its own board field among "
                    f"{len(doc['tasks'])} rows",
                    row=token,
                )
            ],
            notes,
        )
    if len(hits) > 1:
        names = ", ".join(f"{row['id']} ({rule})" for row, rule in hits)
        return (
            [
                _finding(
                    RULE_AMBIGUOUS,
                    f"board row {token!r} matches {len(hits)} rows ({names}); a dispatch "
                    "that cannot be found from one row is not drivable",
                    row=token,
                )
            ],
            notes,
        )
    row, rule = hits[0]
    title, project = milestone_of(doc, row)
    notes.append(
        f"row {row['id']} [{rule}] status={row['status_value'] or '-'} "
        f"milestone={title or '-'} project={project or '-'}"
    )
    notes.append(
        f"move target: board row {row['id']} is currently "
        f"{row['status_value'] or 'unset'}; a hook cannot call MCP and REST exposes no "
        "move-task route, so a hook-side move is out of scope and moving it is the lead's "
        "action"
    )
    if not row["milestone_id"] or not title or not project:
        missing = []
        if not row["milestone_id"]:
            missing.append("milestone id")
        if not title:
            missing.append("milestone title")
        if not project:
            missing.append("project")
        return (
            [
                _finding(
                    RULE_NO_MILESTONE,
                    f"board row {token!r} resolves to {row['id']} ({row['name']}), which "
                    f"is outside a milestone-bearing project: missing {', '.join(missing)}; "
                    "the lead moves it in the UI (no move-task route over REST)",
                    row=row["id"],
                )
            ],
            notes,
        )
    return [], notes


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def check_declaration(
    doc: dict[str, Any], declaration: tuple[str, str, str], source_label: str
) -> tuple[list[Finding], list[str]]:
    """Findings for one ``(source, kind, value)`` declaration against a board."""
    _, kind, value = declaration
    if kind == "internal":
        raise AssertionError("a no-row declaration is checked against the staged paths, not the board")
    token = board.declared_row(value)
    if not token:
        return [_finding(RULE_MISSING, f"the declaration in {source_label} names no row")], []
    return check_row(doc, token)


def classify_no_row(value: str, source_label: str) -> tuple[str, str, list[Finding]]:
    """``(marker, detail, findings)`` for a field the driver read as a no-row declaration.

    ``board.board_declaration`` recognises the whole no-row vocabulary (``none``,
    ``n/a``, ``no row``, ``not applicable``, ``internal``) and returns kind
    ``internal`` for each. Only two exact markers survive, and a marker must be
    the whole remainder of the field: ``none -- housekeeping (really code)`` is
    not ``housekeeping``.

    - ``housekeeping`` -- ``board: none -- housekeeping``
    - ``spike`` -- ``board: none -- spike APEIRON-<EPIC>-SPIKE-<slug>``; a spike
      is accepted on a code-touching commit because AGENTS.md §3 makes it a task
      kind with its own id form, a time limit and a question, ending in evidence
      or a blocker rather than epic closure, so it needs no board row.

    Everything else in that vocabulary, including the retired phrase ``internal
    lane`` and a bare ``none``, is a finding. This runs before any board read, so
    the decision does not depend on a credential or a reachable board.
    """
    if HOUSEKEEPING.match(value):
        return "housekeeping", "", []
    spike = SPIKE_DECLARATION.match(value)
    if spike:
        spike_id = spike.group(1)
        if SPIKE_ID.match(spike_id):
            return "spike", spike_id, []
        return (
            "",
            "",
            [
                _finding(
                    RULE_SPIKE_ID,
                    f"the spike declaration in {source_label} names {spike_id!r}, which is "
                    "not the APEIRON-<EPIC>-SPIKE-<slug> form AGENTS.md §3 fixes for a "
                    "spike; a marker that cannot be read as a task id is not a marker",
                    spike=spike_id,
                )
            ],
        )
    if re.search(r"\binternal\b", value, re.I):
        retired = "the phrase `internal lane`"
    elif board.declared_row(value).lower() == "none":
        retired = "a bare `none`"
    else:
        retired = f"the no-row marker {board.declared_row(value)!r}"
    return (
        "",
        "",
        [
            _finding(
                RULE_NO_ROW_RETIRED,
                f"the declaration in {source_label} is `board: {value}`, and {retired} is "
                "retired: it is not a valid no-row declaration anywhere. The only two are "
                "`board: none -- housekeeping` (a change confined to documentation and "
                "tooling) and `board: none -- spike APEIRON-<EPIC>-SPIKE-<slug>`; every "
                "other commit names a board row with `board: <row-id>`",
            )
        ],
    )


def check_no_row(
    declaration: tuple[str, str, str], paths: list[str], started: float
) -> int:
    """The no-row lane, checked against the staged paths alone.

    Refuses a housekeeping declaration that carries a code path, and reports the
    residual population by name so a commit cannot pass by sitting outside both
    sets. Never reads the board.
    """
    _, _, value = declaration
    source = declaration[0]
    marker, detail, findings = classify_no_row(value, source)
    for finding in findings:
        print(f"lane-declaration: {finding}", file=sys.stderr)
    if findings:
        return 1
    code = code_paths(paths)
    if marker == "housekeeping" and code:
        shown = ", ".join(code[:6]) + (f" (+{len(code) - 6} more)" if len(code) > 6 else "")
        outside = sorted(set(paths) - set(code))
        residual = (
            f"; {len(outside)} staged path(s) sit outside the code set ({', '.join(outside[:4])})"
            if outside
            else ""
        )
        print(
            f"lane-declaration: {RULE_NO_ROW_ON_CODE}: the staged change carries code and "
            f"the declaration in {source} is {value!r}, which names no row: {shown}{residual}. "
            f"Code is {', '.join(CODE_PREFIXES)} and {', '.join(CODE_FILES)}; a change to it "
            "is attributable work and must name a board row that resolves, or a well-formed "
            "spike id. Housekeeping covers documentation and tooling only",
            file=sys.stderr,
        )
        return 1
    if marker == "housekeeping":
        print(
            f"lane-declaration: OK housekeeping declared in {source}: {len(paths)} staged "
            f"path(s), none under the code set"
        )
    else:
        print(
            f"lane-declaration: OK spike {detail} declared in {source}: {len(paths)} staged "
            f"path(s), {len(code)} of them code"
        )
    print(f"lane-declaration: OK declaration {value!r}; no board row to move, no board read required")
    print(f"lane-declaration: {time.time() - started:.2f}s")
    return 0


def check_commit(
    args: argparse.Namespace, texts: list[tuple[str, str]], require: bool
) -> int:
    """Check the declarations in ``texts``; ``require`` demands one of them.

    A git ``pre-commit`` hook cannot read the pending message: at that point
    ``COMMIT_EDITMSG`` still holds the *previous* commit's message (verified,
    2026-10-03, with a probe hook and ``git commit -F``), and the message file
    is written after the hook runs. So ``pre-commit`` checks only what the staged
    change carries and never demands a declaration it cannot see; ``commit-msg``
    is handed the real message file as ``$1`` and is where presence is enforced.
    """
    if board is None:
        print(
            f"lane-declaration: {RULE_DRIVER_MISSING}: {BOARD_DIR / 'board.py'} is not "
            f"importable ({BOARD_IMPORT_ERROR}); the declaration rule is the driver's own, "
            "so it is not re-implemented here",
            file=sys.stderr,
        )
        return 1
    started = time.time()
    try:
        paths = staged_paths()
    except RuntimeError as exc:
        print(f"lane-declaration: {RULE_GIT}: {exc}", file=sys.stderr)
        return 1
    declarations, findings = find_declaration(texts)
    if findings:
        for finding in findings:
            print(f"lane-declaration: {finding}", file=sys.stderr)
        return 1
    if not declarations:
        if require:
            print(
                f"lane-declaration: {RULE_MISSING}: no board declaration in "
                + " or ".join(source for source, _ in texts)
                + f"; {DECLARATION_HINT}",
                file=sys.stderr,
            )
            return 1
        print(
            "lane-declaration: no board declaration in the staged change; the commit "
            "message is checked by .githooks/commit-msg, which is handed the message"
        )
        code = code_paths(paths)
        if code:
            print(
                f"lane-declaration: {len(code)} staged path(s) are code ({', '.join(code[:4])}"
                f"{'+' if len(code) > 4 else ''}); that commit must carry `board: <row-id>` "
                "naming a row that resolves, or `board: none -- spike <id>`"
            )
        return 0

    source, kind, value = declarations[0]
    if kind == "internal":
        return check_no_row(declarations[0], paths, started)

    doc, origin = load_for_check(args)
    if doc is None:
        print(
            f"lane-declaration: {RULE_UNAVAILABLE}: {origin}; no board to resolve "
            f"{board.declared_row(value)!r} against; set OPERATELY_API_KEY, or "
            "APEIRON_HOOKS_BOARD_SOURCE=<schema-1 document> "
            "(`python3 tools/hooks/lane.py capture --out PATH`), or declare "
            "`board: none -- housekeeping` for a documentation/tooling change",
            file=sys.stderr,
        )
        return 1
    more, notes = check_declaration(doc, declarations[0], source)
    for finding in more:
        print(f"lane-declaration: {finding}", file=sys.stderr)
    if more:
        return 1
    for note in notes:
        print(f"lane-declaration: OK {note}")
    print(f"lane-declaration: OK declaration {value!r} in {source}; board {origin}")
    print(f"lane-declaration: {time.time() - started:.2f}s")
    return 0


def mode_staged(args: argparse.Namespace) -> int:
    try:
        texts = [("the staged change", "\n".join(staged_added_lines()))]
    except RuntimeError as exc:
        print(f"lane-declaration: {RULE_GIT}: {exc}", file=sys.stderr)
        return 1
    return check_commit(args, texts, require=False)


def mode_message(args: argparse.Namespace) -> int:
    """The ``commit-msg`` mode: git passes the message file as ``$1``."""
    if not args.path:
        print("lane-declaration: usage: lane.py message <message-file>", file=sys.stderr)
        return 2
    try:
        text = Path(args.path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"lane-declaration: {RULE_GIT}: {args.path}: {exc}", file=sys.stderr)
        return 1
    texts = [("the commit message", text)]
    try:
        texts.append(("the staged change", "\n".join(staged_added_lines())))
    except RuntimeError as exc:
        print(f"lane-declaration: {RULE_GIT}: {exc}", file=sys.stderr)
        return 1
    return check_commit(args, texts, require=True)


def load_for_check(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    """The board for a check, or ``(None, reason)`` when none is reachable.

    A missing board is never silently ignored: a commit that names a row is
    refused, and the caller prints the fix.
    """
    if board is None:
        return None, f"{RULE_DRIVER_MISSING}: {BOARD_DIR / 'board.py'}: {BOARD_IMPORT_ERROR}"
    source = args.board_source or os.environ.get("APEIRON_HOOKS_BOARD_SOURCE", "")
    if source:
        path = Path(source)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            return None, f"{path} does not exist"
        try:
            doc = load_board_documents(str(path), ())
        except board.TransportError as exc:
            return None, f"snapshot {path}: {exc}"
        digest = board.sha256_of(path.read_text(encoding="utf-8"))[:16]
        return doc, f"snapshot {path} sha256:{digest}"
    try:
        doc = load_board_documents(None, board.DEFAULT_PROJECTS)
    except board.TransportError as exc:
        return None, str(exc)
    return doc, "live operately"


def mode_rows(args: argparse.Namespace) -> int:
    doc, origin = load_for_check(args)
    if doc is None:
        print(f"lane-rows: {RULE_UNAVAILABLE}: {origin}", file=sys.stderr)
        return 1
    print(f"lane-rows: {origin}; {len(doc['tasks'])} rows")
    for row in doc["tasks"]:
        title, project = milestone_of(doc, row)
        print(
            f"  {row['id']}\t{row['status_value'] or '-'}\t{title or '-'}\t{project or '-'}\t{row['name']}"
        )
    return 0


def mode_capture(args: argparse.Namespace) -> int:
    """Write the live board as a schema-1 document for an offline hook."""
    doc = load_board_documents(None, board.DEFAULT_PROJECTS)
    raw = {
        "schema": board.SCHEMA,
        "origin": f"{doc['origin']} captured by tools/hooks/lane.py",
        "space": doc["space"],
        "projects": doc["projects"],
        "milestones": [
            {
                "id": m["id"],
                "title": m["title"],
                "status": m["status"],
                "project": {"id": m["project_id"], "name": m["project_name"]}
                if m["project_id"]
                else None,
            }
            for m in doc["milestones"]
        ],
        "tasks": [
            {
                "id": t["id"],
                "name": t["name"],
                "status": {"value": t["status_value"], "label": t["status_label"]},
                "milestone": {"id": t["milestone_id"], "title": t["milestone_title"]}
                if t["milestone_id"]
                else None,
                "description": t["text"],
            }
            for t in doc["tasks"]
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
    reloaded = board.load_document(_source_namespace(str(out)))
    same = (
        len(reloaded["tasks"]) == len(doc["tasks"])
        and all(
            milestone_of(reloaded, a) == milestone_of(doc, b)
            for a, b in zip(reloaded["tasks"], doc["tasks"])
        )
    )
    print(
        f"lane-capture: wrote {out} ({len(doc['tasks'])} tasks, {len(doc['milestones'])} "
        f"milestones); round-trip milestone/project fidelity: {'OK' if same else 'MISMATCH'}"
    )
    return 0 if same else 1


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def _doc(tasks: list[dict[str, Any]], milestones: list[dict[str, Any]]) -> dict[str, Any]:
    return board.normalize_document(
        {"schema": board.SCHEMA, "space": {"id": "s", "name": "s"}, "milestones": milestones, "tasks": tasks}
    )


def control_document() -> dict[str, Any]:
    """A board with one driveable row, one row outside a milestone, one ambiguous pair."""
    return _doc(
        [
            {
                "id": "m1-hooks-abcdef",
                "name": "Hooks gate",
                "status": {"value": "in_progress", "label": "In progress"},
                "milestone": {"id": "m1-plan", "title": "M1: gates"},
                "project": {"id": "p1", "name": "apeiron"},
                "description": "board: APEIRON-HOOKS-W1.",
            },
            {
                "id": "stranded-row-123456",
                "name": "Row outside any milestone",
                "status": {"value": "pending", "label": "Not started"},
                "description": "no milestone",
            },
            {
                "id": "twin-a-111111",
                "name": "Twin lane row",
                "status": {"value": "pending", "label": "Not started"},
                "milestone": {"id": "m1-plan", "title": "M1: gates"},
                "project": {"id": "p1", "name": "apeiron"},
                "description": "",
            },
            {
                "id": "twin-b-222222",
                "name": "Twin lane row",
                "status": {"value": "pending", "label": "Not started"},
                "milestone": {"id": "m1-plan", "title": "M1: gates"},
                "project": {"id": "p1", "name": "apeiron"},
                "description": "",
            },
        ],
        [
            {"id": "m1-plan", "title": "M1: gates", "project": {"id": "p1", "name": "apeiron"}},
            {"id": "m2-empty", "title": "M2: empty", "project": {"id": "p1", "name": "apeiron"}},
        ],
    )


def control_case(
    texts: list[tuple[str, str]], paths: list[str], expect: str, require: bool = True
) -> tuple[bool, str]:
    """Run one seeded case; return (passes, verdict).

    No git and no network: the staged path set is an argument, so each case says
    which commit shape it exercises. A no-row declaration is decided by the
    driver parse plus :func:`classify_no_row` plus the staged paths, exactly as
    :func:`check_no_row` decides it; a row declaration is resolved against the
    control document.
    """
    declarations, findings = find_declaration(texts)
    observed = "ok"
    if findings:
        observed = findings[0]["rule"]
    elif not declarations:
        observed = RULE_MISSING if require else "ok"
    elif declarations[0][1] == "internal":
        marker, _, more = classify_no_row(declarations[0][2], declarations[0][0])
        if more:
            observed = more[0]["rule"]
        elif marker == "housekeeping" and code_paths(paths):
            observed = RULE_NO_ROW_ON_CODE
    else:
        found, _ = check_row(control_document(), board.declared_row(declarations[0][2]))
        observed = found[0]["rule"] if found else "ok"
    if observed == expect:
        return True, f"CONTROL {'GREEN' if observed == 'ok' else 'RED'} [{observed}]"
    return False, f"CONTROL FAIL: expected {expect}, got {observed}"


# The three commit shapes the rule distinguishes.
CODE_STAGED = ["src/table.rs"]
DOCS_STAGED = ["docs/plan.md", "tools/hooks/README.md", ".githooks/commit-msg"]
MIXED_STAGED = ["src/table.rs", "docs/plan.md"]
MANIFEST_STAGED = ["Cargo.toml"]


def run_controls() -> int:
    if board is None:
        print(
            f"lane-declaration: {RULE_DRIVER_MISSING}: {BOARD_DIR / 'board.py'} is not "
            f"importable ({BOARD_IMPORT_ERROR})",
            file=sys.stderr,
        )
        return 2
    def msg(text: str) -> list[tuple[str, str]]:
        return [("the commit message", text)]

    cases = [
        # -- the retired vocabulary, refused by name on every commit ---------
        (
            "retired: `none -- internal lane` on a code commit",
            msg("board: none -- internal lane\n"),
            CODE_STAGED,
            RULE_NO_ROW_RETIRED,
            True,
        ),
        (
            "retired: `none -- internal lane` on a docs-only commit",
            msg("board: none -- internal lane\n"),
            DOCS_STAGED,
            RULE_NO_ROW_RETIRED,
            True,
        ),
        (
            "retired: a bare `board: none` names no marker",
            msg("board: none\n"),
            DOCS_STAGED,
            RULE_NO_ROW_RETIRED,
            True,
        ),
        (
            "retired: `board: n/a` is not one of the two markers",
            msg("board: n/a\n"),
            DOCS_STAGED,
            RULE_NO_ROW_RETIRED,
            True,
        ),
        (
            "retired: `housekeeping` plus a trailing reason is not the marker",
            msg("board: none -- housekeeping (really code)\n"),
            DOCS_STAGED,
            RULE_NO_ROW_RETIRED,
            True,
        ),
        # -- housekeeping is bounded by the staged paths ---------------------
        (
            "refused: housekeeping on a code-touching commit",
            msg("board: none -- housekeeping\n"),
            CODE_STAGED,
            RULE_NO_ROW_ON_CODE,
            True,
        ),
        (
            "refused: housekeeping on a code+docs staged set",
            msg("board: none -- housekeeping\n"),
            MIXED_STAGED,
            RULE_NO_ROW_ON_CODE,
            True,
        ),
        (
            "refused: a crate manifest is code, so housekeeping does not cover it",
            msg("board: none -- housekeeping\n"),
            MANIFEST_STAGED,
            RULE_NO_ROW_ON_CODE,
            True,
        ),
        (
            "accepted: housekeeping on a docs/tooling-only commit",
            msg("board: none -- housekeeping\n"),
            DOCS_STAGED,
            "ok",
            True,
        ),
        # -- spike: the one marker a code commit may use without a row ------
        (
            "accepted: a well-formed spike id on a code-touching commit",
            msg("board: none -- spike APEIRON-HOOKS-SPIKE-probe\n"),
            CODE_STAGED,
            "ok",
            True,
        ),
        (
            "refused: a spike marker whose id is not APEIRON-<EPIC>-SPIKE-<slug>",
            msg("board: none -- spike hooks-spike\n"),
            CODE_STAGED,
            RULE_SPIKE_ID,
            True,
        ),
        # -- the row path ----------------------------------------------------
        (
            "accepted: a code commit naming a row that resolves",
            msg("hooks: install\n\nboard: APEIRON-HOOKS-W1\n"),
            CODE_STAGED,
            "ok",
            True,
        ),
        (
            "refused: a code commit naming a row that is not on the board",
            msg("hooks: install\n\nboard: APEIRON-HOOKS-W9\n"),
            CODE_STAGED,
            RULE_NOT_FOUND,
            True,
        ),
        (
            "seeded: the message with no declaration at all",
            msg("hooks: install\n"),
            CODE_STAGED,
            RULE_MISSING,
            True,
        ),
        (
            "refused: a named row outside a milestone-bearing project",
            msg("board: stranded-row-123456\n"),
            DOCS_STAGED,
            RULE_NO_MILESTONE,
            True,
        ),
        (
            "refused: a token matching two rows is not traceable",
            msg("board: twin-lane-row\n"),
            DOCS_STAGED,
            RULE_AMBIGUOUS,
            True,
        ),
        (
            "refused: two sources naming different rows",
            [
                ("the commit message", "board: APEIRON-HOOKS-W1\n"),
                ("the staged change", "board: m1-hooks-abcdef\n"),
            ],
            CODE_STAGED,
            RULE_CONFLICT,
            True,
        ),
        (
            "accepted: pre-commit does not demand a field it cannot see",
            [("the staged change", "hooks: install\n")],
            CODE_STAGED,
            "ok",
            False,
        ),
        (
            "restore: the accepted row case still passes after every seeded case",
            msg("board: APEIRON-HOOKS-W1\n"),
            CODE_STAGED,
            "ok",
            True,
        ),
    ]
    failures = 0
    print(f"lane-declaration controls ({len(cases)} cases, each seeded defect red by name):")
    for name, texts, paths, expect, require in cases:
        ok, verdict = control_case(texts, paths, expect, require)
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
        print(f"        {verdict}")
    print(f"lane-declaration controls: {len(cases) - failures}/{len(cases)} passed")
    return 0 if failures == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("staged", "message", "capture", "rows", "self-test"))
    parser.add_argument("path", nargs="?", default="", help="the message file, as commit-msg passes it")
    parser.add_argument("--out", help="snapshot path for capture")
    parser.add_argument("--board-source", default="", help="schema-1 board document to read")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "self-test":
        return run_controls()
    if args.mode == "capture":
        if not args.out:
            print("lane-capture: --out PATH is required", file=sys.stderr)
            return 2
        return mode_capture(args)
    if args.mode == "rows":
        return mode_rows(args)
    if args.mode == "message":
        return mode_message(args)
    return mode_staged(args)


if __name__ == "__main__":
    sys.exit(main())