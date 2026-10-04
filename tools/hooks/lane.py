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

1. the live board over REST, when ``OPERATELY_API_KEY`` is in the environment.
   The live board is **enumerated**, never sampled: every route that lists rows is
   read to exhaustion, two enumerations must agree on the row count and the row
   id set, and the routes must agree with each other. An under-read is refused,
   not resolved against. A missing key is a named refusal.
2. ``APEIRON_HOOKS_BOARD_SOURCE`` -- a *stamped* capture (see ``lane.py
   capture``). Accepted only as a capture, never as a bare document: it must
   carry ``fetch.fetched_at`` and ``fetch.task_count``, it must be no older than
   the staged change, and when the key is present its row count must equal a
   fresh enumeration. A snapshot that under-read the board is therefore refused
   rather than authoritative.
3. nothing: a commit that names a row cannot be checked, so it is refused by
   ``board-unavailable`` and the fix is printed. ``housekeeping`` and a
   well-formed spike need no board: neither names a row to move, and the
   no-row decision is taken from the staged paths alone, before any board read,
   so a board outage can neither grant nor withhold it.

A ``board:`` declaration names a row by its **exact** id, or by a plan row id
that exactly one row's own description declares. Name-slug and row-id-prefix
matching are gone: a fuzzy rule is how a declaration for one row resolved to a
different, older row. The named row must not already be ``Done`` or
``Canceled`` (``board-row-closed``), and its id must appear in the committed
change (``board-row-not-in-commit``), so the association is readable from the
commit and not only from the message.

The commit's association with its row is the ``Commit`` object and the
``Commit->Task`` link of
``docs/contracts/commit-task-association.md``. The ``board:`` message field is
one projection of that link; ``lane.py link <rev>`` reads the link back out of a
committed object.

Modes: ``staged`` (the pre-commit half), ``message PATH`` (the commit-msg half),
``capture`` (write a stamped snapshot), ``rows`` (diagnostic), ``link [REV]``
(read a commit back into the object), ``self-test`` (the controls).
"""

from __future__ import annotations

import argparse
import datetime
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
# Enumeration and freshness. An under-read board that resolves nothing is worse
# than no board: it made a valid row read as absent, and the commit that landed
# then declared an older row instead. Every one of these refuses (non-zero).
RULE_FETCH_KEY = "board-fetch-key-missing"
RULE_FETCH_UNAVAILABLE = "board-fetch-unavailable"
RULE_FETCH_TRUNCATED = "board-fetch-page-truncated"
RULE_FETCH_UNSTABLE = "board-fetch-count-unstable"
RULE_FETCH_DISAGREE = "board-fetch-routes-disagree"
RULE_FETCH_PAGING = "board-fetch-paging-unsupported"
RULE_SNAPSHOT_UNSTAMPED = "board-snapshot-unstamped"
RULE_SNAPSHOT_STALE = "board-snapshot-stale"
# Resolution. A row declaration names a row by id; the row must be one this
# commit could serve, and the commit must carry its own association.
RULE_ROW_CLOSED = "board-row-closed"
RULE_ROW_NOT_IN_COMMIT = "board-row-not-in-commit"

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
# Exhaustive enumeration
# ---------------------------------------------------------------------------
#
# The board is enumerated, not sampled. Three measured facts about this instance
# fix the design (probe, `tools/board/board.py` RestClient, `2026-10-04`):
#
#   route                          result
#   ------------------------------  --------------------------------------------------
#   tasks/list?project_id=<id>      95 rows for apeiron, 54 for song, 0 for penglai;
#                                   ``project_id`` is required; payload keys are
#                                   ``['tasks']`` only
#   spaces/list_tasks?space_id=..   14 rows, none of them in apeiron's 95
#   projects/list_tasks             HTTP 404 ``Query not found``
#   projects/list_milestone_tasks   26 apeiron rows, a strict subset of the 95
#
# So the milestone walk cannot enumerate the board: it saw 26 of 95, the space
# listing sees a different 14, and their union -- the pre-fix ``fetch_live``
# path -- produced the **77 rows** this hook resolved against, which is why a
# live row read as ``board-row-not-found``.
#
# No route on this instance supports paging: ``limit``, ``cursor``, ``offset``,
# ``per_page``, ``take``, ``count``, ``after``, ``first``, ``page`` are rejected
# as ``Unknown input field``, and ``page_size`` / ``starting_after`` /
# ``items_per_page`` / ``max_results`` pass schema validation and then answer
# HTTP 500. There is no cursor to follow and no declared total, so completeness
# cannot be read off a single response. It is established from three
# independent signals, and the report names which ones carried the argument:
#
#   1. *cap*: a response shorter than the requested page size, or a declared
#      total the accumulated rows meet;
#   2. *stability*: two independent enumerations return the same id set and the
#      same count;
#   3. *cross-route*: every row the milestone walk returns is a row the
#      project listing also returns.
#
# A route with no cap signal and no stability signal is refused rather than
# trusted. A route that returns a page at the requested size with neither a
# cursor nor a total is refused as truncated.

REQUESTED_PAGE = 500
MAX_PAGES = 64
# Envelope keys a paged listing may use for its continuation and its total.
CURSOR_KEYS = (
    "next_cursor",
    "next",
    "cursor",
    "continuation",
    "next_page",
    "next_page_token",
    "page_token",
)
TOTAL_KEYS = ("total", "total_count", "totalCount", "count", "task_count", "total_tasks")
# The input field that carries a continuation, keyed by the envelope key that
# announced it. An unknown key is refused rather than guessed at.
CURSOR_PARAM_FOR_KEY = {
    "next_cursor": "starting_after",
    "next": "cursor",
    "cursor": "starting_after",
    "continuation": "starting_after",
    "next_page": "page",
    "next_page_token": "starting_after",
    "page_token": "starting_after",
}
# Accepted-but-unimplemented inputs, in the order they are tried. A 500 on one of
# these is recorded as "this instance does not page", not swallowed. The
# continuation inputs are not probed: a continuation is followed only if a route
# returns one, and then the request field comes from CURSOR_PARAM_FOR_KEY.
PAGE_SIZE_PARAMS = ("page_size", "items_per_page", "per_page", "limit", "max_results")

NO_PAGING = "no paging input accepted"


class FetchError(RuntimeError):
    """A named enumeration refusal; ``rule`` is what the hook prints."""

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail


def _continuation(payload: dict[str, Any]) -> tuple[str, str]:
    """``(key, value)`` of the continuation token, or ``("", "")``."""
    for key in CURSOR_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return key, value
        if isinstance(value, dict):
            for inner in CURSOR_KEYS:
                if value.get(inner):
                    return key, str(value[inner])
    nested = payload.get("pagination") or payload.get("paging") or {}
    if isinstance(nested, dict):
        for key in CURSOR_KEYS:
            if isinstance(nested.get(key), str) and nested[key]:
                return key, str(nested[key])
    return "", ""


def _declared_total(payload: dict[str, Any]) -> int | None:
    for key in TOTAL_KEYS:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
    nested = payload.get("pagination") or payload.get("paging") or {}
    if isinstance(nested, dict):
        for key in TOTAL_KEYS:
            value = nested.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None


# Which page-size input, if any, each collection's schema accepts, and why the
# rest were rejected. Keyed by the collection in the route (``tasks``,
# ``projects``, ``spaces``): the instance validates inputs per collection, so
# probing once per collection is enough, and it is what keeps a per-commit
# double enumeration inside the hook budget (measured 47.2s probing every route
# against 3.4s probing each collection once, 2026-10-04). Caching cannot hide a
# truncation: the refusal is the page-at-the-requested-size rule, which holds
# whether or not a page-size input was sent.
_PAGE_INPUT: dict[str, tuple[str, list[str]]] = {}


def _page_input(client: Any, path: str, params: dict[str, Any]) -> tuple[str | None, list[str]]:
    """The page-size input this collection accepts, probing once per collection."""
    collection = path.split("/", 1)[0]
    cached = _PAGE_INPUT.get(collection)
    if cached is not None:
        return cached
    winner: str | None = None
    rejected: list[str] = []
    for candidate in PAGE_SIZE_PARAMS:
        try:
            client.call(path, **{**params, candidate: REQUESTED_PAGE})
        except board.TransportError as exc:
            rejected.append(f"{candidate} rejected ({str(exc)[-40:]})")
            continue
        winner = candidate
        break
    _PAGE_INPUT[collection] = (winner, rejected)
    return winner, rejected


def paginate(
    client: Any, path: str, collection: str, **fixed: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every row of one listing route. Returns ``(rows, report)``.

    A page carrying a continuation is followed to exhaustion; a page at the
    requested size with no continuation and no declared total is refused as
    truncated; a repeated continuation is refused as a cursor loop. The report
    records the paging signal this instance offered, which is measured, not
    assumed.
    """
    params = dict(fixed)
    winner, rejected = _page_input(client, path, params)
    paging: str
    payload: dict[str, Any] | None = None
    if winner is None:
        paging = NO_PAGING + (f"; {'; '.join(rejected)}" if rejected else "")
        try:
            payload = client.call(path, **params)
        except board.TransportError as exc:
            raise FetchError(RULE_FETCH_UNAVAILABLE, f"{path}: {exc}") from None
    else:
        paging = f"{winner}={REQUESTED_PAGE}"
        try:
            payload = client.call(path, **{**params, winner: REQUESTED_PAGE})
        except board.TransportError as exc:
            raise FetchError(RULE_FETCH_UNAVAILABLE, f"{path}: {exc}") from None

    rows: list[dict[str, Any]] = []
    pages = 0
    cursors: set[str] = set()
    total: int | None = None
    while True:
        pages += 1
        if pages > MAX_PAGES:
            raise FetchError(
                RULE_FETCH_PAGING, f"{path}: {pages} pages without exhaustion (cap {MAX_PAGES})"
            )
        if pages > 1:
            try:
                payload = client.call(path, **params)
            except board.TransportError as exc:
                raise FetchError(RULE_FETCH_UNAVAILABLE, f"{path}: {exc.detail}") from None
        page = [row for row in (payload.get(collection) or []) if isinstance(row, dict)]
        rows.extend(page)
        if total is None:
            total = _declared_total(payload)
        cursor_key, cursor = _continuation(payload)
        if cursor:
            if cursor in cursors:
                raise FetchError(
                    RULE_FETCH_PAGING, f"{path}: continuation {cursor!r} repeated (cursor loop)"
                )
            cursors.add(cursor)
            param = CURSOR_PARAM_FOR_KEY.get(cursor_key)
            if param is None:
                raise FetchError(
                    RULE_FETCH_PAGING,
                    f"{path}: continuation key {cursor_key!r} has no known request field",
                )
            params = {**fixed, param: cursor}
            continue
        if total is not None:
            if len(rows) < total:
                raise FetchError(
                    RULE_FETCH_TRUNCATED,
                    f"{path}: the route declares total={total} and returned {len(rows)} "
                    f"row(s) after {pages} page(s) with no continuation",
                )
            signal = f"total={total}"
        elif len(rows) >= REQUESTED_PAGE:
            raise FetchError(
                RULE_FETCH_TRUNCATED,
                f"{path}: a page of {len(rows)} rows is at the requested page size "
                f"({REQUESTED_PAGE}) with no continuation and no declared total, so the "
                "response may be hiding rows",
            )
        else:
            signal = f"{len(rows)} rows < requested page size {REQUESTED_PAGE}"
        return rows, {"path": path, "pages": pages, "paging": paging, "signal": signal}


def exhaustive_document(
    client: Any,
    space_id: str = board.DEFAULT_SPACE_ID,
    projects: Iterable[str] = board.DEFAULT_PROJECTS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The whole board, from every route that enumerates rows.

    ``projects/list_milestone_tasks`` is a cross-route check, not the authority:
    measured, it sees 26 of apeiron's 95 rows. ``spaces/list_tasks`` adds the
    rows nothing else enumerates. The union is the document, every row is
    attributed to the project whose listing returned it, and a row the walk
    returns that the listing does not is refused.
    """
    wanted = {p.lower() for p in projects}
    chosen: dict[str, dict[str, Any]] = {}
    for project in client.call("projects/list").get("projects") or []:
        name = (project.get("name") or "").lower()
        if not wanted or name in wanted:
            chosen[project["id"]] = {
                "id": project["id"],
                "name": project.get("name") or project["id"],
            }
    routes: list[dict[str, Any]] = []
    by_project: dict[str, list[dict[str, Any]]] = {}
    for project_id, project in chosen.items():
        rows, report = paginate(client, "tasks/list", "tasks", project_id=project_id)
        routes.append({**report, "project": project["name"], "rows": len(rows)})
        by_project[project_id] = rows

    raw_tasks: dict[str, dict[str, Any]] = {}
    for project_id, rows in by_project.items():
        project = chosen[project_id]
        for task in rows:
            # The listing filter *is* the attribution: `tasks/list` answers
            # `project: null` on every row, so the project is set from the query
            # that produced the row. An explicit ``None`` is overwritten, not
            # merely defaulted -- `setdefault` would keep the null and leave the
            # row unattributed, which is `unattributed-task` at the barrier.
            if not task.get("project"):
                task["project"] = project
            raw_tasks[task["id"]] = task

    walked = 0
    milestones_by_project: dict[str, list[dict[str, Any]]] = {}
    for project_id, project in chosen.items():
        milestones = list(
            client.call("projects/list_milestones", project_id=project_id).get("milestones") or []
        )
        milestones_by_project[project_id] = milestones
        for milestone in milestones:
            rows, report = paginate(
                client, "projects/list_milestone_tasks", "tasks", milestone_id=milestone["id"]
            )
            routes.append(
                {
                    **report,
                    "milestone": milestone.get("title") or milestone["id"],
                    "rows": len(rows),
                }
            )
            walked += len(rows)
            for task in rows:
                listed = raw_tasks.get(task["id"])
                if listed is None:
                    raise FetchError(
                        RULE_FETCH_DISAGREE,
                        f"milestone {milestone.get('title')!r} lists task {task['id']}, which "
                        f"tasks/list for project {project['name']!r} does not return; the two "
                        "routes disagree and the population cannot be closed",
                    )
                listed["milestone"] = {
                    "id": milestone.get("id"),
                    "title": milestone.get("title"),
                }
                if not listed.get("project"):
                    listed["project"] = project

    raw_milestones: list[dict[str, Any]] = [
        {
            "id": milestone.get("id"),
            "title": milestone.get("title"),
            "status": milestone.get("status"),
            "project": chosen[project_id],
        }
        for project_id, milestones in milestones_by_project.items()
        for milestone in milestones
    ]

    space_rows: list[dict[str, Any]] = []
    try:
        space_rows, report = paginate(client, "spaces/list_tasks", "tasks", space_id=space_id)
        routes.append({**report, "scope": "space", "rows": len(space_rows)})
    except FetchError as exc:
        # A space-level listing that will not page is a route absence, not a
        # red board: the project routes above already closed the population. It
        # is recorded, because a row only this route sees is then unreachable.
        routes.append({"path": "spaces/list_tasks", "error": exc.rule})
        space_rows = []
    for task in space_rows:
        raw_tasks.setdefault(task["id"], task)

    document = {
        "schema": board.SCHEMA,
        "origin": "live:operately (lane.py exhaustive enumeration)",
        "space": {"id": space_id, "name": "sanguine"},
        "projects": list(chosen.values()),
        "milestones": raw_milestones,
        "tasks": list(raw_tasks.values()),
        "notes": list(board.LIVE_COVERAGE_NOTES)
        + [
            f"enumeration: projects={len(chosen)} milestone-walk rows={walked} "
            f"union rows={len(raw_tasks)}; routes="
            + ", ".join(
                f"{r.get('project') or r.get('milestone') or r.get('scope') or r['path']}"
                f"={r.get('rows', r.get('error'))}"
                for r in routes
            ),
        ],
    }
    normalized = board.normalize_document(document)
    board.attribute_projects(normalized)
    return normalized, {"routes": routes, "task_count": len(normalized["tasks"])}


def enumerated_board(
    client: Any,
    space_id: str = board.DEFAULT_SPACE_ID,
    projects: Iterable[str] = board.DEFAULT_PROJECTS,
    repeats: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The board enumerated twice, with the two id sets and counts compared.

    Two independent reads are the stability signal: a count that changes between
    them, or an id set that differs, is refused. A single read is never accepted
    as a closed enumeration.
    """
    if repeats < 2:
        raise FetchError(
            RULE_FETCH_UNSTABLE, f"enumeration requires at least 2 reads, asked for {repeats}"
        )
    document: dict[str, Any] | None = None
    report: dict[str, Any] = {}
    seen: list[tuple[int, frozenset[str]]] = []
    for attempt in range(repeats):
        document, report = exhaustive_document(client, space_id, projects)
        seen.append((len(document["tasks"]), frozenset(t["id"] for t in document["tasks"])))
    counts = {count for count, _ in seen}
    if len(counts) != 1:
        raise FetchError(
            RULE_FETCH_UNSTABLE,
            f"{repeats} enumerations returned different row counts ({sorted(counts)}); the "
            "board is being written under the fetch and no single read can be authority",
        )
    if len({ids for _, ids in seen}) != 1:
        first, second = seen[0][1], seen[1][1]
        raise FetchError(
            RULE_FETCH_UNSTABLE,
            f"{repeats} enumerations returned different row id sets "
            f"(+{len(second - first)}/-{len(first - second)} between reads); the board moved "
            "under the fetch",
        )
    report = {**report, "reads": repeats, "task_count": seen[0][0]}
    assert document is not None
    return document, report


def client_from_environment() -> Any:
    """``board.RestClient`` when ``OPERATELY_API_KEY`` is set, else a refusal.

    The key is read from the environment only (``board.RestClient`` reads
    ``OPERATELY_API_KEY`` itself and scrubs it from every message); a missing key
    is a named refusal, never a silent empty board.
    """
    if not os.environ.get(board.KEY_ENV, "").strip():
        raise FetchError(
            RULE_FETCH_KEY,
            f"{board.KEY_ENV} is not set in the environment, so the live board cannot be "
            "enumerated; export it, or use a stamped snapshot through "
            "APEIRON_HOOKS_BOARD_SOURCE",
        )
    return board.RestClient()


# ---------------------------------------------------------------------------
# Board resolution
# ---------------------------------------------------------------------------


def load_board_documents(source: str) -> dict[str, Any]:
    """A snapshot read through the driver's own parser.

    Raises ``board.TransportError`` when the snapshot is not a schema-1
    document. The live arm is ``enumerated_board``, which enumerates rather than
    samples; this function is the snapshot arm only.
    """
    return board.load_document(_source_namespace(source))


def _source_namespace(source: str) -> argparse.Namespace:
    """The driver's own parsed arguments for a snapshot read.

    Built from ``board.build_parser`` so every attribute ``load_document``
    reads is present; a hand-built namespace would drift from the driver.
    """
    return board.build_parser().parse_args(["list", "--source", source])


def parse_epoch(stamp: str) -> float:
    """An ISO-8601 timestamp as a POSIX epoch, or ``0.0`` when unparseable."""
    text = (stamp or "").strip()
    if not text:
        return 0.0
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def staged_change_epoch(root: Path | None = None) -> float:
    """When the staged change was last written: the git index mtime.

    ``git add`` writes the index, so a capture older than this index predates
    the change it would be asked to resolve -- the staleness comparison the
    snapshot arm needs, at one stat instead of one stat per staged path.
    """
    try:
        return (git_dir(root) / "index").stat().st_mtime
    except (RuntimeError, OSError):
        return 0.0


def snapshot_stamp(path: Path) -> dict[str, Any]:
    """The stamp a snapshot must carry, or ``board-snapshot-unstamped``.

    A schema-1 document written before this rule existed carries no capture
    time and no row count, so it cannot be compared against a staged change and
    cannot be checked against a live count. Such a document is refused by name
    rather than read as authority.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FetchError(RULE_SNAPSHOT_UNSTAMPED, f"snapshot {path}: {exc}") from None
    if not isinstance(raw, dict):
        raise FetchError(RULE_SNAPSHOT_UNSTAMPED, f"snapshot {path}: not a JSON object")
    fetch = raw.get("fetch") if isinstance(raw.get("fetch"), dict) else {}
    fetched_at = fetch.get("fetched_at")
    count = fetch.get("task_count")
    rows = raw.get("tasks") if isinstance(raw.get("tasks"), list) else []
    problems: list[str] = []
    if not parse_epoch(str(fetched_at or "")):
        problems.append("fetch.fetched_at is absent or not an ISO-8601 timestamp")
    if isinstance(count, bool) or not isinstance(count, int):
        problems.append("fetch.task_count is absent or not an integer")
    elif count != len(rows):
        problems.append(f"fetch.task_count={count} but the document holds {len(rows)} row(s)")
    if problems:
        raise FetchError(
            RULE_SNAPSHOT_UNSTAMPED,
            f"snapshot {path} is not a stamped capture: {'; '.join(problems)}. Write one "
            "with `python3 tools/hooks/lane.py capture --out PATH`; a capture carries "
            "`fetch.fetched_at` (ISO-8601 UTC), `fetch.task_count` and the fetch signals",
        )
    return {
        "fetched_at": str(fetched_at),
        "epoch": parse_epoch(str(fetched_at)),
        "task_count": int(count),
        "reads": fetch.get("reads"),
        "signals": str(fetch.get("signals") or ""),
    }


def match_rows(doc: dict[str, Any], token: str) -> list[tuple[dict[str, Any], str]]:
    """Rows the declared token names, each with the rule that matched it.

    Two exact rules, both from the driver's own vocabulary
    (``board.board_declaration`` / ``board.plan_ids``): the token equals a row
    id, or the token equals a plan row id that exactly one row's own description
    declares. Name-slug and row-id-prefix matching were **removed**: a token is
    an id, and a fuzzy rule is how a declaration for one row resolved to a
    different, older row.
    """
    low = token.strip().lower()
    hits: list[tuple[dict[str, Any], str]] = []
    for row in doc["tasks"]:
        rule = ""
        if (row["id"] or "").lower() == low:
            rule = "row id"
        elif any(plan_id.lower() == low for plan_id in row["plan_ids"]):
            rule = "plan row id declared by this row's own description"
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


def check_row(doc: dict[str, Any], token: str) -> tuple[list[Finding], list[str], str]:
    """Findings, informational notes and the resolved row id for a declared row.

    Three refusals, each load-bearing:

    * ``board-row-not-found`` -- the exact id names no row. Measured: the
      pre-fix fetch resolved against 77 of 95 rows, so a live row read as
      absent;
    * ``board-row-ambiguous`` -- the token is a plan row id declared by more
      than one row;
    * ``board-row-closed`` -- the named row is Done or Canceled. This is the
      adopted plausibility rule: a commit names the row it moves, and a row
      that is already closed cannot be the row a change is still landing
      against, so an older closed row cannot stand in for the live one. A row
      in any other state (the driver's ``actionable``, ``other``) is accepted.
    """
    hits = match_rows(doc, token)
    notes: list[str] = []
    if not hits:
        return (
            [
                _finding(
                    RULE_NOT_FOUND,
                    f"board row {token!r} is not on the board: no row matches it by exact id "
                    f"or as a plan row declared by exactly one row among {len(doc['tasks'])} "
                    "rows; a declaration must carry the row's exact id",
                    row=token,
                )
            ],
            notes,
            "",
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
            "",
        )
    row, rule = hits[0]
    title, project = milestone_of(doc, row)
    bucket = board.status_bucket(row)
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
    if bucket in ("done", "canceled"):
        return (
            [
                _finding(
                    RULE_ROW_CLOSED,
                    f"board row {token!r} resolves to {row['id']} ({row['name']}), which reads "
                    f"{row['status_value'] or bucket}: a closed or canceled row is not the row a "
                    "change can be landing against, so declaring it substitutes a finished row "
                    "for the live one. Name the open row this commit serves, or, for a commit "
                    "that carries no attributable work, `board: none -- housekeeping`",
                    row=row["id"],
                )
            ],
            notes,
            row["id"],
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
            row["id"],
        )
    return [], notes, row["id"]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def check_declaration(
    doc: dict[str, Any], declaration: tuple[str, str, str], source_label: str
) -> tuple[list[Finding], list[str], str]:
    """Findings, notes and the resolved row id for one declaration."""
    _, kind, value = declaration
    if kind == "internal":
        raise AssertionError("a no-row declaration is checked against the staged paths, not the board")
    token = board.declared_row(value)
    if not token:
        return [_finding(RULE_MISSING, f"the declaration in {source_label} names no row")], [], ""
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


def row_carried_in_change(row_id: str, staged_text: str) -> bool:
    """True when the resolved row id appears in the committed change.

    The single place the in-commit rule is decided, shared by ``check_commit``
    and the control case, so a control cannot agree with the code by reading it
    twice and deciding it twice.
    """
    return bool(row_id) and row_id in staged_text


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
    staged_text = "\n".join(text for name, text in texts if name == "the staged change")
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

    try:
        doc, origin = load_for_check(args)
    except FetchError as exc:
        print(f"lane-declaration: {exc}", file=sys.stderr)
        return 1
    if doc is None:
        print(
            f"lane-declaration: {RULE_UNAVAILABLE}: {origin}; no board to resolve "
            f"{board.declared_row(value)!r} against; set OPERATELY_API_KEY, or "
            "APEIRON_HOOKS_BOARD_SOURCE=<stamped capture> "
            "(`python3 tools/hooks/lane.py capture --out PATH`), or declare "
            "`board: none -- housekeeping` for a documentation/tooling change",
            file=sys.stderr,
        )
        return 1
    more, notes, row_id = check_declaration(doc, declarations[0], source)
    for finding in more:
        print(f"lane-declaration: {finding}", file=sys.stderr)
    if more:
        return 1
    # The commit must carry its own association: the resolved row id appears in
    # the committed content, so the link is readable from the commit object
    # alone and not only from the message field that declared it.
    if row_id and not row_carried_in_change(row_id, staged_text):
        print(
            f"lane-declaration: {RULE_ROW_NOT_IN_COMMIT}: the declaration resolves to row "
            f"{row_id}, which appears nowhere in the staged change; a commit whose association "
            "lives only in its message cannot be read back from the commit. Add the row id to "
            "an added line (a receipt or plan row naming it is the usual place), or declare "
            "`board: none -- housekeeping` if the change carries no attributable work",
            file=sys.stderr,
        )
        return 1
    for note in notes:
        print(f"lane-declaration: OK {note}")
    print(f"lane-declaration: OK declaration {value!r} in {source}; board {origin}")
    lane, named, _ = lane_of(kind, value)
    print(
        "lane-commit: "
        + json.dumps(
            project_commit(
                row_id=named,
                lane=lane,
                paths=paths,
                message="\n".join(text for name, text in texts if name == "the commit message"),
            ),
            separators=(",", ":"),
        )
    )
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


def capture_freshness(stamp: dict[str, Any], staged: float, path: Path) -> None:
    """Refuse a capture that predates the staged change.

    ``git add`` writes the index, so an index mtime newer than the capture means
    the capture was taken before the change it would resolve. Raises
    ``board-snapshot-stale``.
    """
    if staged and stamp["epoch"] < staged:
        raise FetchError(
            RULE_SNAPSHOT_STALE,
            f"snapshot {path} was captured at {stamp['fetched_at']} ({stamp['epoch']:.0f}) and "
            f"the staged change was last written at {staged:.0f}: a capture older than the "
            "change it would resolve cannot be authority. Re-capture with `python3 "
            f"tools/hooks/lane.py capture --out {path}` after staging",
        )


def load_for_check(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    """The board for a check, or ``(None, reason)`` when none is reachable.

    The live board is the default and is enumerated, never sampled. A snapshot
    is accepted only as a *stamped capture*: it must carry
    ``fetch.fetched_at`` and ``fetch.task_count``, it must be no older than the
    staged change (the git index mtime), and when ``OPERATELY_API_KEY`` is in the
    environment its row count must equal a fresh enumeration. A snapshot that
    under-read the board therefore cannot be authority, which is the defect this
    rule closes: the pre-fix capture held 77 of 95 rows and refused a live row.

    Raises ``FetchError`` for a missing key, an unstamped capture, a stale
    capture and a count that disagrees with the live board.
    """
    if board is None:
        return None, f"{RULE_DRIVER_MISSING}: {BOARD_DIR / 'board.py'}: {BOARD_IMPORT_ERROR}"
    source = args.board_source or os.environ.get("APEIRON_HOOKS_BOARD_SOURCE", "")
    if not source:
        document, report = enumerated_board(
            client_from_environment(), board.DEFAULT_SPACE_ID, list(board.DEFAULT_PROJECTS)
        )
        routes = "; ".join(
            f"{r.get('project') or r.get('milestone') or r.get('scope') or r['path']}={r.get('rows', r.get('error'))}"
            for r in report["routes"]
        )
        return document, (
            f"live operately, enumerated {report['reads']}x over "
            f"{len(report['routes'])} route(s) [{routes}]: {report['task_count']} rows, "
            f"no paging input accepted on this instance; closure from {report['reads']}-read "
            "stability and cross-route agreement"
        )
    path = Path(source)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return None, f"{path} does not exist"
    stamp = snapshot_stamp(path)
    capture_freshness(stamp, staged_change_epoch(), path)
    document = load_board_documents(str(path))
    digest = board.sha256_of(path.read_text(encoding="utf-8"))[:16]
    origin = (
        f"snapshot {path} sha256:{digest} fetched_at={stamp['fetched_at']} "
        f"tasks={stamp['task_count']}"
    )
    if os.environ.get(board.KEY_ENV, "").strip():
        _, live = enumerated_board(
            client_from_environment(), board.DEFAULT_SPACE_ID, list(board.DEFAULT_PROJECTS)
        )
        if live["task_count"] != stamp["task_count"]:
            raise FetchError(
                RULE_FETCH_UNSTABLE,
                f"snapshot {path} holds {stamp['task_count']} row(s) while the live board now "
                f"enumerates {live['task_count']}; a capture that under-reads is refused as "
                "authority rather than resolved against",
            )
        origin += f"; live count {live['task_count']} agrees"
    else:
        origin += f"; no {board.KEY_ENV}, so the count could not be cross-checked live"
    return document, origin


def mode_rows(args: argparse.Namespace) -> int:
    try:
        doc, origin = load_for_check(args)
    except FetchError as exc:
        print(f"lane-rows: {exc}", file=sys.stderr)
        return 1
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


def capture_document(doc: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Project the enumerated board into a stamped schema-1 document.

    The ``fetch`` block is what makes a capture authority rather than a file:
    ``fetched_at`` is the ISO-8601 UTC capture time the commit hook compares
    against the staged change, ``task_count`` is the row count it compares
    against the live board, and ``signals`` names the completeness signals that
    carried the argument on this instance.
    """
    routes = report.get("routes") or []
    return {
        "schema": board.SCHEMA,
        "origin": f"{doc['origin']} captured by tools/hooks/lane.py",
        "fetch": {
            "fetched_at": now_iso(),
            "epoch": int(time.time()),
            "task_count": len(doc["tasks"]),
            "milestone_count": len(doc["milestones"]),
            "project_count": len(doc["projects"]),
            "reads": report.get("reads"),
            "ids_sha256": board.sha256_of(
                "\n".join(sorted(t["id"] for t in doc["tasks"]))
            ),
            "signals": (
                f"reads={report.get('reads')} "
                f"routes={len(routes)} "
                f"cross_route_rows={sum(1 for r in routes if 'milestone' in r)} "
                f"paging={sorted({r['paging'] for r in routes if r.get('paging')})}"
            ),
            "routes": routes,
        },
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
                "project": {"id": t["project_id"], "name": t["project_name"]}
                if t["project_id"]
                else None,
                "description": t["text"],
            }
            for t in doc["tasks"]
        ],
    }


def mode_capture(args: argparse.Namespace) -> int:
    """Enumerate the live board and write it as a stamped capture.

    Two enumerations are compared before anything is written: an under-read that
    passed quietly is the defect this replaces.
    """
    try:
        doc, report = enumerated_board(
            client_from_environment(), board.DEFAULT_SPACE_ID, list(board.DEFAULT_PROJECTS)
        )
    except FetchError as exc:
        print(f"lane-capture: {exc}", file=sys.stderr)
        return 1
    raw = capture_document(doc, report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
    reloaded = board.load_document(_source_namespace(str(out)))
    stamp = snapshot_stamp(out)
    same = (
        len(reloaded["tasks"]) == len(doc["tasks"])
        and all(
            milestone_of(reloaded, a) == milestone_of(doc, b)
            for a, b in zip(reloaded["tasks"], doc["tasks"])
        )
    )
    print(
        f"lane-capture: wrote {out} ({len(doc['tasks'])} tasks, {len(doc['milestones'])} "
        f"milestones, {report['reads']} enumerations agreeing); round-trip milestone/project "
        f"fidelity: {'OK' if same else 'MISMATCH'}"
    )
    print(
        f"lane-capture: stamp fetched_at={stamp['fetched_at']} task_count={stamp['task_count']} "
        f"ids_sha256={raw['fetch']['ids_sha256'][:16]} signals={raw['fetch']['signals']}"
    )
    return 0 if same else 1


# ---------------------------------------------------------------------------
# The Commit object and the Commit -> Task link
# ---------------------------------------------------------------------------
#
# The board ontology is the schema-1 normalized document of
# ``tools/board/board.py`` (``space``, ``projects``, ``milestones``, ``tasks``,
# ``writes``), documented in ``tools/board/README.md``. Its commit-side
# extension is the ``Commit`` object below and the link that carries a commit to
# the row it serves, defined in
# ``docs/contracts/commit-task-association.md``.
#
#   Commit        identity: the commit sha (pre-commit: the index digest)
#   sha           git object id of the commit; "" until the commit exists
#   tree          git tree object id of the commit
#   author        git author identity as ``git var GIT_AUTHOR_IDENT`` reports it
#   timestamp     author date, ISO-8601
#   row           the declared board row's exact id, resolved; "" in a no-row lane
#   lane          row | housekeeping | spike
#   paths         the committed paths
#
#   link Commit->Task: ``{"type": "Commit->Task", "source": "commit:<sha|tree>",
#   "target": "task:<row>"}``. The ``board:`` message field is one *projection*
#   of the link; the link is the object, and it is read back from the commit.

COMMIT_LANES = ("row", "housekeeping", "spike")
LINK_TYPE = "Commit->Task"


def commit_identity(root: Path | None = None) -> tuple[str, str]:
    """``(author, timestamp)`` as git would record them for the next commit."""
    try:
        ident = git("var", "GIT_AUTHOR_IDENT", root=root).strip()
    except RuntimeError:
        return "", ""
    parts = re.match(r"^(.*?)\s*<([^>]*)>\s*(\d+)\s*([+-]\d{4})?$", ident)
    if not parts:
        return ident, ""
    stamp = int(parts.group(3))
    offset = parts.group(4) or "+0000"
    sign = 1 if offset[0] == "+" else -1
    hours, minutes = int(offset[1:3]), int(offset[3:5])
    moment = datetime.datetime.fromtimestamp(
        stamp - sign * (hours * 3600 + minutes * 60), tz=datetime.timezone.utc
    )
    return f"{parts.group(1).strip()} <{parts.group(2).strip()}>", moment.isoformat()


def index_digest(root: Path | None = None) -> str:
    """The staged pre-image identity: the sha256 of the git index.

    A commit object does not exist while ``commit-msg`` runs, so the prospective
    commit is identified by what it will contain -- the index digest -- and by
    its staged diff digest. ``mode_link`` reads the durable identity (the commit
    sha) back from the committed object.
    """
    import hashlib

    try:
        return hashlib.sha256((git_dir(root) / "index").read_bytes()).hexdigest()
    except (RuntimeError, OSError):
        return ""


def project_commit(
    row_id: str = "",
    lane: str = "row",
    paths: Iterable[str] = (),
    message: str = "",
    sha: str = "",
    tree: str = "",
    author: str = "",
    timestamp: str = "",
    root: Path | None = None,
) -> dict[str, Any]:
    """The ``Commit`` object for a prospective or committed change."""
    if lane not in COMMIT_LANES:
        raise ValueError(f"lane must be one of {COMMIT_LANES}, got {lane!r}")
    if not sha:
        author, timestamp = commit_identity(root)
        tree = tree or index_digest(root)
    try:
        diff_digest = board.sha256_of(git("diff", "--cached", "-U0", "--no-color", root=root))
    except RuntimeError:
        diff_digest = ""
    return {
        "schema": board.SCHEMA,
        "commits": [
            {
                "sha": sha,
                "tree": tree,
                "author": author,
                "timestamp": timestamp,
                "row": row_id,
                "lane": lane,
                "paths": list(paths),
                "message_sha256": board.sha256_of(message or ""),
                "index_sha256": index_digest(root),
                "staged_diff_sha256": diff_digest,
            }
        ],
        "links": [
            {
                "type": LINK_TYPE,
                "source": f"commit:{sha or tree}",
                "target": f"task:{row_id}",
                "lane": lane,
            }
        ]
        if row_id
        else [],
    }


def commit_links(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every ``Commit->Task`` link a schema-1 document carries, read back.

    The reader is the round-trip: the object is written into the document and
    this function reads the link out of the document, so a link that only
    exists in the writer is not a link.
    """
    out: list[dict[str, Any]] = []
    commits = [c for c in document.get("commits") or [] if isinstance(c, dict)]
    index: dict[str, dict[str, Any]] = {}
    for commit in commits:
        for identity in (commit.get("sha"), commit.get("tree")):
            if identity:
                index[str(identity)] = commit
    for link in document.get("links") or []:
        if not isinstance(link, dict) or link.get("type") != LINK_TYPE:
            continue
        identity = str(link.get("source", "")).split(":", 1)[-1]
        commit = index.get(identity, {})
        out.append({**link, "row": commit.get("row", "")})
    return out


def lane_of(kind: str, value: str) -> tuple[str, str, list[Finding]]:
    """``(lane, row_id, findings)`` for one declaration, with no board read.

    The single place the lane is decided, shared by ``check_commit`` and
    ``mode_link`` so a commit cannot be projected with a row and read back as
    ``housekeeping``.
    """
    if kind == "internal":
        marker, _detail, findings = classify_no_row(value, "the commit message")
        return marker or "row", "", findings
    return "row", board.declared_row(value), []


def mode_link(args: argparse.Namespace) -> int:
    """Read a committed change back into the ``Commit`` object and its link.

    ``lane.py link`` with no argument reads ``HEAD``. This is the projection the
    ``board:`` message field is one view of: the row id is read out of the
    committed message, the identity out of the commit object.
    """
    revision = args.path or "HEAD"
    try:
        fields = git(
            "show",
            "-s",
            "--format=%H%x1f%T%x1f%an <%ae>%x1f%aI%x1f%B",
            revision,
        ).split("\x1f")
    except RuntimeError as exc:
        print(f"lane-link: {RULE_GIT}: {exc}", file=sys.stderr)
        return 1
    if len(fields) < 5:
        print(f"lane-link: {RULE_GIT}: {revision} is not a commit", file=sys.stderr)
        return 1
    sha, tree, author, timestamp, message = fields[:5]
    paths = [line.strip() for line in git("show", "--name-only", "--format=", revision).splitlines() if line.strip()]
    declarations, findings = find_declaration([("the commit message", message)])
    if findings:
        for finding in findings:
            print(f"lane-link: {finding}", file=sys.stderr)
        return 1
    row_id = ""
    lane = "housekeeping"
    for _source, kind, value in declarations:
        lane, named, refused = lane_of(kind, value)
        if refused:
            print(f"lane-link: {refused[0]}", file=sys.stderr)
            return 1
        row_id = row_id or named
    if row_id and lane != "row":
        print(f"lane-link: {RULE_ROW_CLOSED}: lane {lane!r} carries a row", file=sys.stderr)
        return 1
    projected = project_commit(
        row_id=row_id, lane=lane, paths=paths, message=message, sha=sha, tree=tree,
        author=author, timestamp=timestamp,
    )
    print(json.dumps(projected, indent=1))
    links = commit_links(projected)
    for link in links:
        print(f"lane-link: {link['source']} --{link['type']}--> {link['target']} (lane={link['lane']})")
    if row_id and not links:
        print(f"lane-link: the commit declares row {row_id} but the object carries no link", file=sys.stderr)
        return 1
    if not row_id:
        print(f"lane-link: lane={lane}, no row to associate (a no-row declaration)")
    return 0


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def _doc(tasks: list[dict[str, Any]], milestones: list[dict[str, Any]]) -> dict[str, Any]:
    return board.normalize_document(
        {"schema": board.SCHEMA, "space": {"id": "s", "name": "s"}, "milestones": milestones, "tasks": tasks}
    )


def control_document() -> dict[str, Any]:
    """A board with one driveable row, one row outside a milestone, one ambiguous pair.

    The descriptions carry their plan row ids in backticks, which is where
    ``board.normalize_task`` reads them: a commit may name a row by its exact
    Operately id or by a plan row id exactly one row declares.
    """
    return _doc(
        [
            {
                "id": "m1-hooks-abcdef",
                "name": "Hooks gate",
                "status": {"value": "in_progress", "label": "In progress"},
                "milestone": {"id": "m1-plan", "title": "M1: gates"},
                "project": {"id": "p1", "name": "apeiron"},
                "description": "Plan row `APEIRON-HOOKS-W1`.",
            },
            {
                "id": "stranded-row-123456",
                "name": "Row outside any milestone",
                "status": {"value": "pending", "label": "Not started"},
                "description": "no milestone",
            },
            {
                "id": "closed-row-999999",
                "name": "Already closed row",
                "status": {"value": "done", "label": "Done"},
                "milestone": {"id": "m1-plan", "title": "M1: gates"},
                "project": {"id": "p1", "name": "apeiron"},
                "description": "Receipt `docs/evidence/closed.md`.",
            },
            {
                "id": "twin-a-111111",
                "name": "Twin lane row",
                "status": {"value": "pending", "label": "Not started"},
                "milestone": {"id": "m1-plan", "title": "M1: gates"},
                "project": {"id": "p1", "name": "apeiron"},
                "description": "Plan row `APEIRON-TWIN-W1`.",
            },
            {
                "id": "twin-b-222222",
                "name": "Twin lane row",
                "status": {"value": "pending", "label": "Not started"},
                "milestone": {"id": "m1-plan", "title": "M1: gates"},
                "project": {"id": "p1", "name": "apeiron"},
                "description": "Plan row `APEIRON-TWIN-W1`.",
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
    control document by exact id and must appear in the staged text, exactly as
    :func:`check_commit` decides it.
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
        found, _, row_id = check_row(control_document(), board.declared_row(declarations[0][2]))
        if found:
            observed = found[0]["rule"]
        elif not row_carried_in_change(
            row_id, "\n".join(text for name, text in texts if name == "the staged change")
        ):
            observed = RULE_ROW_NOT_IN_COMMIT
    if observed == expect:
        return True, f"CONTROL {'GREEN' if observed == 'ok' else 'RED'} [{observed}]"
    return False, f"CONTROL FAIL: expected {expect}, got {observed}"


# ---------------------------------------------------------------------------
# Enumeration controls: a scripted client, no network
# ---------------------------------------------------------------------------


class StubClient:
    """A ``RestClient`` whose route answers are scripted.

    ``script`` maps a route path to a list of payloads returned in order (the
    last one repeats). ``fail`` maps a route path to a ``TransportError`` raised
    instead. This is how the truncation, cursor-loop, cross-route disagreement
    and read-instability refusals are exercised without the board.
    """

    def __init__(self, script: dict[str, list[dict[str, Any]]], fail: dict[str, str] | None = None) -> None:
        self.script = script
        self.fail = fail or {}
        self.calls: list[str] = []

    def call(self, path: str, **params: Any) -> dict[str, Any]:
        self.calls.append(path)
        if path in self.fail:
            raise board.TransportError(self.fail[path])
        pages = self.script.get(path)
        if not pages:
            return {}
        index = min(self.calls.count(path) - 1, len(pages) - 1)
        return pages[index]


def stub_task(task_id: str, milestone: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "id": task_id,
        "name": task_id,
        "status": {"value": "in_progress", "label": "In progress"},
        "description": "",
        "milestone": {"id": milestone, "title": milestone} if milestone else None,
        **extra,
    }


def fetch_case(
    name: str,
    client: StubClient,
    expect: str,
    repeats: int = 2,
    projects: Iterable[str] = ("apeiron",),
) -> tuple[bool, str]:
    """Run one enumeration case; return (passes, verdict)."""
    try:
        document, report = enumerated_board(client, "sanguine-test", projects, repeats)
        observed = f"ok tasks={len(document['tasks'])} reads={report['reads']}"
    except FetchError as exc:
        observed = exc.rule
    if observed.startswith(expect):
        return True, f"CONTROL {'GREEN' if observed.startswith('ok') else 'RED'} [{observed}]"
    return False, f"CONTROL FAIL: expected {expect}, got {observed}"


def link_round_trip_case() -> tuple[bool, str]:
    """The ``Commit -> Task`` link must read back out of the object it was written into.

    Hermetic: the identity is passed in, because the prospective identity (the
    index digest) needs a repository and this control must not. The index-digest
    path is exercised by the live hook run, not here.
    """
    identity = "t" * 64
    projected = project_commit(
        row_id="row-0001",
        lane="row",
        paths=["src/table.rs"],
        message="board: row-0001",
        tree=identity,
        root=None,
    )
    document = {**projected, "schema": board.SCHEMA}
    links = commit_links(document)
    if len(links) != 1:
        return False, f"CONTROL FAIL round-trip: expected 1 link, read back {len(links)}"
    if links[0]["target"] != "task:row-0001" or links[0]["row"] != "row-0001":
        return False, f"CONTROL FAIL round-trip: link reads back as {links[0]}"
    # The negative: a different row must read back differently, or the read-back
    # would pass for any value at all.
    mutated = project_commit(
        row_id="row-0002",
        lane="row",
        paths=["src/table.rs"],
        message="board: row-0002",
        tree=identity,
        root=None,
    )
    other = commit_links({**mutated, "schema": board.SCHEMA})
    if other and other[0]["target"] == links[0]["target"]:
        return False, "CONTROL FAIL round-trip: a different row read back as the same link"
    # A no-row lane carries no link, and says so.
    bare = commit_links({**project_commit(row_id="", lane="housekeeping", root=None), "schema": board.SCHEMA})
    if bare:
        return False, f"CONTROL FAIL round-trip: a housekeeping commit carries {len(bare)} link(s)"
    # The lane is decided in one place: a row declaration reads back as `row`
    # with a link, a no-row marker as its own lane with none.
    for kind, value, expect_lane, expect_row in (
        ("row", "row-0001", "row", "row-0001"),
        ("internal", "none -- housekeeping", "housekeeping", ""),
        ("internal", "none -- spike APEIRON-HOOKS-SPIKE-probe", "spike", ""),
    ):
        lane, named, refused = lane_of(kind, value)
        if refused or lane != expect_lane or named != expect_row:
            return False, (
                f"CONTROL FAIL round-trip: declaration {value!r} reads back as "
                f"lane={lane!r} row={named!r}, expected lane={expect_lane!r} row={expect_row!r}"
            )
        projected_lane = project_commit(row_id=named, lane=lane, root=None)["commits"][0]
        if projected_lane["lane"] != lane or projected_lane["row"] != named:
            return False, f"CONTROL FAIL round-trip: the object disagrees with its declaration: {projected_lane}"
    return True, "CONTROL GREEN [Commit->Task link reads back from the object, and is row-specific]"


def contract_parity_case() -> tuple[bool, str]:
    """Every property and rule the contract declares must exist in this module.

    Without this the contract document is prose about a different set of fields
    and rules than the code enforces: the ``Commit`` property table is read from
    the document and compared with what ``project_commit`` writes, and every rule
    the contract's rules table names must be a rule constant here.
    """
    path = REPO_ROOT / "docs" / "contracts" / "commit-task-association.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"CONTROL FAIL contract-parity: {path}: {exc}"
    projected = project_commit(row_id="row-0001", lane="row", root=None)
    written = set(projected["commits"][0]) | set(projected["links"][0])
    properties = table_columns_after(text, "### `Commit`", r"[a-z][a-z0-9_]*")
    rules = table_columns_after(text, "## Rules the association is checked against", r"[a-z][a-z0-9_-]*")
    missing = [name for name in properties if name not in written]
    if missing:
        return False, f"CONTROL FAIL contract-parity: the contract declares {missing}, not written"
    declared = {
        value
        for key, value in globals().items()
        if key.startswith("RULE_") and isinstance(value, str)
    }
    unknown = [rule for rule in rules if rule not in declared]
    if unknown:
        return False, f"CONTROL FAIL contract-parity: the contract names rules {unknown} this module does not define"
    return True, (
        f"CONTROL GREEN [contract declares {len(properties)} Commit properties, all written; "
        f"{len(rules)} rule names, all defined]"
    )


def table_columns_after(text: str, heading: str, name: str = r"[a-z][a-z0-9_]*") -> set[str]:
    """First-column names of the first markdown table below ``heading``.

    The heading must match, so a second table elsewhere in the document (the
    measured-routes table, the gate table) cannot be mistaken for the property
    table.
    """
    lines = text.splitlines()
    names: set[str] = set()
    table: list[list[str]] = []
    inside = False

    def flush(rows: list[list[str]]) -> None:
        for cells in rows[1:]:  # rows[0] is the header
            names.add(cells[0].strip().strip("` "))

    for line in lines:
        if line.startswith("#") and heading in line:
            inside = True
            continue
        if not inside:
            continue
        if line.startswith("## "):
            break
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                table.append(cells)
            continue
        if table:
            flush(table)
            table = []
    if table:
        flush(table)
    return {column for column in names if re.fullmatch(name, column)}


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
        # -- the row path: exact id, or a plan row exactly one row declares ---
        (
            "accepted: a code commit naming the exact row id, carried in the change",
            [
                ("the commit message", "hooks: install\n\nboard: m1-hooks-abcdef\n"),
                ("the staged change", "receipt for m1-hooks-abcdef\n"),
            ],
            CODE_STAGED,
            "ok",
            True,
        ),
        (
            "accepted: a plan row id declared by exactly one row's description",
            [
                ("the commit message", "hooks: install\n\nboard: APEIRON-HOOKS-W1\n"),
                ("the staged change", "receipt for m1-hooks-abcdef\n"),
            ],
            CODE_STAGED,
            "ok",
            True,
        ),
        (
            "refused: a code commit naming a row that is not on the board",
            [
                ("the commit message", "hooks: install\n\nboard: m1-hooks-999999\n"),
                ("the staged change", "receipt for m1-hooks-999999\n"),
            ],
            CODE_STAGED,
            RULE_NOT_FOUND,
            True,
        ),
        (
            "refused: a name slug matching a row (the pre-fix fuzzy rules are gone)",
            [
                ("the commit message", "hooks: install\n\nboard: twin-lane-row\n"),
                ("the staged change", "receipt for m1-hooks-abcdef\n"),
            ],
            CODE_STAGED,
            RULE_NOT_FOUND,
            True,
        ),
        (
            "refused: the resolved row id appears nowhere in the committed change",
            [
                ("the commit message", "hooks: install\n\nboard: m1-hooks-abcdef\n"),
                ("the staged change", "no row id in this change\n"),
            ],
            CODE_STAGED,
            RULE_ROW_NOT_IN_COMMIT,
            True,
        ),
        (
            "refused: a Done row cannot be the row a change lands against",
            [
                ("the commit message", "hooks: install\n\nboard: closed-row-999999\n"),
                ("the staged change", "receipt for closed-row-999999\n"),
            ],
            CODE_STAGED,
            RULE_ROW_CLOSED,
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
            [
                ("the commit message", "board: stranded-row-123456\n"),
                ("the staged change", "receipt for stranded-row-123456\n"),
            ],
            DOCS_STAGED,
            RULE_NO_MILESTONE,
            True,
        ),
        (
            "refused: a plan row id declared by two rows is not traceable",
            [
                ("the commit message", "board: APEIRON-TWIN-W1\n"),
                ("the staged change", "receipt for APEIRON-TWIN-W1\n"),
            ],
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
            [
                ("the commit message", "board: m1-hooks-abcdef\n"),
                ("the staged change", "receipt for m1-hooks-abcdef\n"),
            ],
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

    print("lane-enumeration controls (a scripted client, no network):")
    project = {"id": "p-apeiron", "name": "apeiron"}

    def routes(script: dict[str, list[dict[str, Any]]], fail: dict[str, str] | None = None) -> StubClient:
        base: dict[str, list[dict[str, Any]]] = {
            "projects/list": [{"projects": [project]}],
            "projects/list_milestones": [{"milestones": []}],
        }
        base.update(script)
        return StubClient(base, fail)

    truncation = routes(
        {"tasks/list": [{"tasks": [stub_task(f"row-{i:05d}") for i in range(REQUESTED_PAGE)]}]}
    )
    declared_short = routes(
        {"tasks/list": [{"tasks": [stub_task("row-0001")], "total": 9}]}
    )
    looping = routes(
        {
            "tasks/list": [
                {"tasks": [stub_task("row-0001")], "next_cursor": "c1"},
                {"tasks": [stub_task("row-0002")], "next_cursor": "c1"},
            ]
        }
    )
    unserved = routes(
        {
            "tasks/list": [{"tasks": [stub_task("row-0001")]}],
            "projects/list_milestones": [{"milestones": [{"id": "m1", "title": "M1", "status": "a"}]}],
            "projects/list_milestone_tasks": [{"tasks": [stub_task("row-9999", "m1")]}],
        }
    )
    unstable = routes(
        {
            "tasks/list": [
                {"tasks": [stub_task("row-0001"), stub_task("row-0002")]},
                {
                    "tasks": [
                        stub_task("row-0001"),
                        stub_task("row-0002"),
                        stub_task("row-0003"),
                    ]
                },
            ]
        }
    )
    unreachable = routes(
        {"tasks/list": [{"tasks": [stub_task("row-0001")]}]},
        fail={"spaces/list_tasks": "spaces/list_tasks: HTTP 500"},
    )
    listed = [stub_task(f"row-{i:04d}", "m1") for i in range(3)]
    agreeing = routes(
        {
            "tasks/list": [{"tasks": listed}],
            "projects/list_milestones": [{"milestones": [{"id": "m1", "title": "M1", "status": "a"}]}],
            "projects/list_milestone_tasks": [{"tasks": listed[:2]}],
            "spaces/list_tasks": [{"tasks": [stub_task("space-0001")]}],
        }
    )
    fetch_cases = [
        ("accepted: every route agrees over two enumerations", agreeing, "ok"),
        (
            "refused: a page at the requested size with no continuation is a truncation",
            truncation,
            RULE_FETCH_TRUNCATED,
        ),
        (
            "refused: a declared total the pages do not meet is a truncation",
            declared_short,
            RULE_FETCH_TRUNCATED,
        ),
        ("refused: a repeated continuation is refused, not looped", looping, RULE_FETCH_PAGING),
        (
            "refused: a milestone lists a row the project listing does not return",
            unserved,
            RULE_FETCH_DISAGREE,
        ),
        (
            "refused: a count that changes between two enumerations",
            unstable,
            RULE_FETCH_UNSTABLE,
        ),
    ]
    for name, client, expect in fetch_cases:
        ok, verdict = fetch_case(name, client, expect)
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
        print(f"        {verdict}")

    ok, verdict = fetch_case("refused: a single read", agreeing, RULE_FETCH_UNSTABLE, repeats=1)
    failures += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'}  refused: a single enumeration is not a closed population")
    print(f"        {verdict}")

    # An unavailable auxiliary route is recorded, not fatal: the project routes
    # already closed the population. The control states that both ways.
    ok, verdict = fetch_case("accepted: a space route that will not answer", unreachable, "ok")
    failures += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'}  accepted: an unavailable space route is recorded, not fatal")
    print(f"        {verdict}")

    print("lane-capture controls (a capture must carry a stamp, and be fresh):")
    missing_key = os.environ.pop(board.KEY_ENV, None)
    try:
        try:
            client_from_environment()
            ok, verdict = False, f"CONTROL FAIL [{RULE_FETCH_KEY}]: a missing key returned a client"
        except FetchError as exc:
            ok = exc.rule == RULE_FETCH_KEY
            verdict = f"CONTROL RED [{exc.rule}]"
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  refused: a missing OPERATELY_API_KEY is a named refusal")
        print(f"        {verdict}")

        tmp = Path(os.environ.get("TMPDIR", "/tmp"))
        for name, payload, expect in (
            (
                "refused: a capture carrying no fetch stamp",
                {"schema": board.SCHEMA, "space": {}, "milestones": [], "tasks": []},
                RULE_SNAPSHOT_UNSTAMPED,
            ),
            (
                "refused: a capture whose stamped count disagrees with its rows",
                {
                    "schema": board.SCHEMA,
                    "fetch": {"fetched_at": now_iso(), "task_count": 7},
                    "space": {},
                    "milestones": [],
                    "tasks": [],
                },
                RULE_SNAPSHOT_UNSTAMPED,
            ),
        ):
            path = tmp / f"lane-control-{board.sha256_of(name)[:12]}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            try:
                snapshot_stamp(path)
                observed = "ok"
            except FetchError as exc:
                observed = exc.rule
            path.unlink(missing_ok=True)
            ok = observed == expect
            failures += 0 if ok else 1
            print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
            print(f"        {'CONTROL RED' if ok else 'CONTROL FAIL'} [{observed}]")

        # Staleness: the same stamp is fresh against an older index and stale
        # against a newer one, so the control can see the comparison decide.
        staged = staged_change_epoch() or time.time()
        moment = datetime.datetime.fromtimestamp(staged, tz=datetime.timezone.utc)
        for name, delta, expect in (
            ("refused: a capture older than the staged change", 3600, RULE_SNAPSHOT_STALE),
            ("accepted: a capture newer than the staged change", -60, "ok"),
        ):
            stamp = {
                "fetched_at": (moment - datetime.timedelta(seconds=delta))
                .replace(microsecond=0)
                .isoformat(),
                "epoch": staged - delta,
                "task_count": 0,
            }
            try:
                capture_freshness(stamp, staged, tmp / "capture.json")
                observed = "ok"
            except FetchError as exc:
                observed = exc.rule
            ok = observed == expect
            failures += 0 if ok else 1
            print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
            print(f"        {'CONTROL RED' if observed != 'ok' else 'CONTROL GREEN'} [{observed}]")
    finally:
        if missing_key is not None:
            os.environ[board.KEY_ENV] = missing_key

    print("lane-ontology controls (the Commit -> Task link and the contract):")
    for name, runner in (
        ("the Commit->Task link round-trips through the document", link_round_trip_case),
        ("every contract property is a field the projector writes", contract_parity_case),
    ):
        ok, verdict = runner()
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
        print(f"        {verdict}")

    total = len(cases) + 15
    print(f"lane-declaration controls: {total - failures}/{total} passed")
    return 0 if failures == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "mode", choices=("staged", "message", "capture", "rows", "link", "self-test")
    )
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
    if args.mode == "link":
        return mode_link(args)
    if args.mode == "message":
        return mode_message(args)
    return mode_staged(args)


if __name__ == "__main__":
    sys.exit(main())