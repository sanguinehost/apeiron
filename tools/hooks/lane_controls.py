#!/usr/bin/env python3
"""lane_controls: the self-test cases behind `lane.py self-test`.

Every case is a seeded defect that must redden by name, with no
git and no network: the staged path set and the board are arguments. Part of
``tools/hooks/lane.py``. ``contract_parity_case`` reads the ``RULE_*`` names from
this module's globals, so every rule the driver names is imported here.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from typing import Iterable
from lane_core import (  # noqa: F401
    BOARD_DIR,
    BOARD_IMPORT_ERROR,
    FetchError,
    REPO_ROOT,
    RULE_AMBIGUOUS,
    RULE_CONFLICT,
    RULE_DRIVER_MISSING,
    RULE_FETCH_DISAGREE,
    RULE_FETCH_KEY,
    RULE_FETCH_PAGING,
    RULE_FETCH_TRUNCATED,
    RULE_FETCH_UNAVAILABLE,
    RULE_FETCH_UNSTABLE,
    RULE_GIT,
    RULE_MISSING,
    RULE_NOT_FOUND,
    RULE_NO_MILESTONE,
    RULE_NO_ROW_ON_CODE,
    RULE_NO_ROW_RETIRED,
    RULE_ROW_CLOSED,
    RULE_ROW_NOT_IN_COMMIT,
    RULE_SNAPSHOT_STALE,
    RULE_SNAPSHOT_UNSTAMPED,
    RULE_SPIKE_ID,
    RULE_UNAVAILABLE,
    board,
    check_row,
    classify_no_row,
    code_paths,
    commit_links,
    find_declaration,
    lane_of,
    now_iso,
    project_commit,
    row_carried_in_change,
    snapshot_stamp,
    staged_change_epoch,
)
from lane_fetch import (  # noqa: F401
    REQUESTED_PAGE,
    client_from_environment,
    enumerated_board,
)
from lane_modes import (  # noqa: F401
    capture_freshness,
)

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


class ChurnClient(StubClient):
    """A client whose row count grows on every read: no two reads agree.

    This is the test double for the quorum budget. The fetch must re-enumerate
    until the budget expires and then refuse ``board-fetch-count-unstable`` --
    the fix must not have turned a board that never quiesces into an accepted
    later read. The count grows by one row per ``tasks/list`` call, so
    ``read[n] != read[n-1]`` forever.
    """

    def __init__(self) -> None:
        super().__init__({})

    def call(self, path: str, **params: Any) -> dict[str, Any]:
        self.calls.append(path)
        if path == "projects/list":
            return {"projects": [{"id": "p-apeiron", "name": "apeiron"}]}
        if path == "projects/list_milestones":
            return {"milestones": []}
        if path == "tasks/list":
            count = self.calls.count("tasks/list")
            return {"tasks": [stub_task(f"row-{i:05d}") for i in range(count)]}
        return {}


def fetch_case(
    name: str,
    client: StubClient,
    expect: str,
    repeats: int = 2,
    projects: Iterable[str] = ("apeiron",),
    budget: float | None = None,
) -> tuple[bool, str]:
    """Run one enumeration case; return (passes, verdict)."""
    try:
        document, report = enumerated_board(client, "sanguine-test", projects, repeats, budget)
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
            "accepted: a count that changes once, then quiesces, is retried to agreement",
            unstable,
            "ok tasks=3",
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

    # The negative control the quiesce retry must not remove: a board that never
    # has two consecutive agreeing reads is still refused, after the budget,
    # with board-fetch-count-unstable. The budget is a fraction of a second so
    # the control is hermetic and instant.
    ok, verdict = fetch_case(
        "refused: a board that never quiesces expires the budget",
        ChurnClient(),
        RULE_FETCH_UNSTABLE,
        budget=0.05,
    )
    failures += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'}  refused: a board that never quiesces still refuses by name")
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

    total = len(cases) + 16
    print(f"lane-declaration controls: {total - failures}/{total} passed")
    return 0 if failures == 0 else 2
