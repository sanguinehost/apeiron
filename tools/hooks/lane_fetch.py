#!/usr/bin/env python3
# Row: `apeiron-harness-lane-land-CnFUhFHaC4TnQIA1dG3yRF` (APEIRON-HARNESS-LANE-FETCH-INSTALL).
"""lane_fetch: exhaustive enumeration of the live board.

Read to exhaustion, never sampled: every route that lists rows is
enumerated, two enumerations must agree on the row count and the row id set, and
the routes must agree with each other. An under-read is refused, not resolved
against. Part of ``tools/hooks/lane.py``.
"""

from __future__ import annotations

import os
import time
from typing import Any
from typing import Iterable
from lane_core import (
    COMMITMSG_BUDGET_ENV,
    QUORUM_HEADROOM,
    FetchError,
    RULE_FETCH_DISAGREE,
    RULE_FETCH_KEY,
    RULE_FETCH_PAGING,
    RULE_FETCH_TRUNCATED,
    RULE_FETCH_UNAVAILABLE,
    RULE_FETCH_UNSTABLE,
    board,
    commitmsg_budget,
)

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


def _unstable_after_budget(
    reads: list[tuple[int, frozenset[str]]], budget: float
) -> FetchError:
    """The named refusal for a board that never quiesced inside the budget.

    The refusal is the same one the two-read comparison always raised: the board
    was written under the fetch and no single read can be authority. It is not
    degraded to the later read -- that is the whole defect this rule exists to
    refuse.
    """
    counts = sorted({count for count, _ in reads})
    if len(counts) > 12:
        counts_text = ", ".join(str(count) for count in counts[:8]) + f", ... +{len(counts) - 8} more"
    else:
        counts_text = ", ".join(str(count) for count in counts)
    if len(counts) > 1:
        what = f"different row counts ({counts_text})"
    else:
        first, second = reads[-2][1], reads[-1][1]
        what = (
            f"the same row count ({counts[0]}) but different row id sets "
            f"(+{len(second - first)}/-{len(first - second)} between the last two reads)"
        )
    headroom = min(QUORUM_HEADROOM, budget / 2.0)
    return FetchError(
        RULE_FETCH_UNSTABLE,
        f"{len(reads)} enumerations over a {budget:g}s budget returned {what}; the board is "
        "being written under the fetch and no single read can be authority. The budget "
        f"({COMMITMSG_BUDGET_ENV}={budget:g}s less {headroom:g}s headroom) expired while the "
        "board kept moving, so the fetch refuses and never accepts the later read. For a "
        "board that will not quiesce, write a stamped capture "
        "(`python3 tools/hooks/lane.py capture --out PATH`) and set APEIRON_HOOKS_BOARD_SOURCE",
    )


def enumerated_board(
    client: Any,
    space_id: str = board.DEFAULT_SPACE_ID,
    projects: Iterable[str] = board.DEFAULT_PROJECTS,
    repeats: int = 2,
    budget: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The board enumerated to quiescence: two *consecutive* reads must agree.

    The board is never resolved against a partially-read snapshot. A single read
    is never accepted. Two consecutive enumerations that return the same row
    count and the same row id set are the closure signal; while a writer is
    mid-flight the first read may see fewer rows than the next, so the fetch
    re-enumerates until two consecutive reads agree. This is bounded by a
    declared wall-clock budget -- ``budget`` (seconds), or
    ``APEIRON_HOOKS_COMMITMSG_BUDGET`` (default 90 s) when it is ``None``. If the
    budget expires while the board keeps moving, the fetch raises
    ``board-fetch-count-unstable`` with the observed counts; it does not degrade
    to "take the later read". A busy board therefore has a deterministic path:
    ``python3 tools/hooks/lane.py capture --out PATH`` writes a stamped capture
    and ``APEIRON_HOOKS_BOARD_SOURCE`` makes it the source.

    ``repeats`` is the minimum number of reads; it must be at least 2, so a
    single-read call is still refused.
    """
    if repeats < 2:
        raise FetchError(
            RULE_FETCH_UNSTABLE, f"enumeration requires at least 2 reads, asked for {repeats}"
        )
    if budget is None:
        budget = commitmsg_budget()
    deadline = time.monotonic() + max(budget - QUORUM_HEADROOM, budget / 2.0)
    reads: list[tuple[int, frozenset[str]]] = []
    document: dict[str, Any] | None = None
    report: dict[str, Any] = {}
    while True:
        document, report = exhaustive_document(client, space_id, projects)
        reads.append((len(document["tasks"]), frozenset(t["id"] for t in document["tasks"])))
        # The budget is checked only once the two-read quorum exists: a board
        # with a tiny test budget must still get its second read to decide
        # agreement, or an agreeing board would be refused unread.
        if len(reads) >= repeats and reads[-1] == reads[-2]:
            break
        if len(reads) >= repeats and time.monotonic() >= deadline:
            raise _unstable_after_budget(reads, budget)
    report = {**report, "reads": len(reads), "task_count": reads[-1][0]}
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
