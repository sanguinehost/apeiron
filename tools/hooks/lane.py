#!/usr/bin/env python3
"""The per-commit board-declaration requirement (AGENTS.md §9.3).

A dispatch names its row. Every commit carries the declaration, either in the
message or in the staged change:

    board: <row-id>
    board: none -- internal lane

The field is parsed by ``tools/board/board.py`` itself -- ``board_declaration``,
``declared_row`` and ``slug`` -- so the hook enforces the driver's rule rather
than a second spelling of it. A named row must exist on the board and sit in a
milestone of a project: a row outside a milestone cannot be selected, ordered or
driven, which is the ``unattributed-task`` failure class the driver already
reports.

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
   ``board-unavailable`` and the fix is printed. A commit that declares
   ``none -- internal lane`` needs no board and passes without one.

Modes: ``staged`` (the pre-commit half), ``message PATH`` (the commit-msg half),
``capture`` (write a snapshot), ``rows`` (diagnostic), ``self-test`` (the
controls).
"""

from __future__ import annotations

import argparse
import json
import os
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

DECLARATION_HINT = (
    "the commit must carry a labelled field on its own line, in the message or "
    "in the staged change:\n"
    "    board: <row-id>\n"
    "    board: none -- internal lane"
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
    token = board.declared_row(value)
    if kind == "internal":
        return [], [f"internal lane declared in {source_label}: no board row to move"]
    if not token:
        return [_finding(RULE_MISSING, f"the declaration in {source_label} names no row")], []
    return check_row(doc, token)


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
        return 0

    source, kind, value = declarations[0]
    if kind == "internal":
        # An internal lane has no row to resolve, so it needs no board: a commit
        # that says so must not depend on a reachable board or a credential.
        print(f"lane-declaration: OK internal lane declared in {source}: no board row to move")
        print(f"lane-declaration: OK declaration {value!r}; no board read required")
        print(f"lane-declaration: {time.time() - started:.2f}s")
        return 0

    doc, origin = load_for_check(args)
    if doc is None:
        print(
            f"lane-declaration: {RULE_UNAVAILABLE}: {origin}; no board to resolve "
            f"{board.declared_row(value)!r} against; set OPERATELY_API_KEY, or "
            "APEIRON_HOOKS_BOARD_SOURCE=<schema-1 document> "
            "(`python3 tools/hooks/lane.py capture --out PATH`), or declare "
            "`board: none -- internal lane`",
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
    texts: list[tuple[str, str]], expect: str | None, require: bool = True
) -> tuple[bool, str]:
    """Run one seeded case against the control document; return (passes, verdict)."""
    doc = control_document()
    declarations, findings = find_declaration(texts)
    if findings:
        observed = findings[0]["rule"]
    elif not declarations:
        observed = RULE_MISSING if require else "ok"
    else:
        more, _ = check_declaration(doc, declarations[0], declarations[0][0])
        observed = more[0]["rule"] if more else "ok"
    if observed == (expect or "ok"):
        return True, f"CONTROL {'GREEN' if observed == 'ok' else 'RED'} [{observed}]"
    return False, f"CONTROL FAIL: expected {expect or 'ok'}, got {observed}"


def run_controls() -> int:
    if board is None:
        print(
            f"lane-declaration: {RULE_DRIVER_MISSING}: {BOARD_DIR / 'board.py'} is not "
            f"importable ({BOARD_IMPORT_ERROR})",
            file=sys.stderr,
        )
        return 2
    cases = [
        (
            "baseline: a declared row that exists in a milestone-bearing project",
            [("the commit message", "hooks: install\n\nboard: APEIRON-HOOKS-W1\n")],
            "ok",
            True,
        ),
        (
            "seeded: the message with no declaration at all",
            [("the commit message", "hooks: install\n")],
            RULE_MISSING,
            True,
        ),
        (
            "seeded: a named row that is not on the board",
            [("the commit message", "hooks: install\n\nboard: APEIRON-HOOKS-W9\n")],
            RULE_NOT_FOUND,
            True,
        ),
        (
            "seeded: a named row outside a milestone-bearing project",
            [("the commit message", "board: stranded-row-123456\n")],
            RULE_NO_MILESTONE,
            True,
        ),
        (
            "seeded: an internal lane needs no board row",
            [("the commit message", "board: none -- internal lane\n")],
            "ok",
            True,
        ),
        (
            "seeded: a token matching two rows is not traceable",
            [("the commit message", "board: twin-lane-row\n")],
            RULE_AMBIGUOUS,
            True,
        ),
        (
            "seeded: two sources naming different rows",
            [
                ("the commit message", "board: APEIRON-HOOKS-W1\n"),
                ("the staged change", "board: m1-hooks-abcdef\n"),
            ],
            RULE_CONFLICT,
            True,
        ),
        (
            "seeded: a declaration in the staged change only is not missing",
            [("the staged change", "hooks: install\n")],
            "ok",
            False,
        ),
        (
            "restore: the baseline still passes after the seeded cases",
            [("the commit message", "board: APEIRON-HOOKS-W1\n")],
            "ok",
            True,
        ),
    ]
    failures = 0
    print("lane-declaration controls (each seeded defect red by name):")
    for name, texts, expect, require in cases:
        ok, verdict = control_case(texts, expect, require)
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