#!/usr/bin/env python3
"""board_check: reconciliation, the contract check and rendering.

"--check" compares the board's non-closed rows against the named
source of truth and classifies every row as ``mirrored``, ``internal-by-design``
or ``missing``; the seeded defects below are the mutations that must each redden
it by name. Part of ``tools/board/board.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any
from typing import Iterable
from board_core import (
    BOARD_DECLARATION,
    DEFAULT_PROJECTS,
    Finding,
    PROJECT_PLANS,
    RULE_DONE_NO_RECEIPT,
    RULE_DRIFT_BOARD_ROW,
    RULE_DRIFT_PLAN_ROW,
    RULE_DRIFT_STALE,
    RULE_DUE_WITHOUT_FACT,
    RULE_MILESTONE_EMPTY,
    RULE_NO_BOARD_DECLARATION,
    RULE_NO_GATE,
    RULE_NO_PATHS,
    RULE_PLAN_UNDECLARED,
    RULE_UNATTRIBUTED,
    RULE_UNKNOWN_MILESTONE,
    RULE_WRITE_NOT_LANDED,
    _finding,
    board_declaration,
    dated_facts,
    declared_row,
    due_date_of,
    rules_of,
    slug,
)
from board_model import (
    MIN_SLUG_MATCH,
    load_plan,
    non_closed_tasks,
    normalize_task,
    plan_from_document,
    select_tasks,
    status_bucket,
)
from board_fixtures import (
    SONG,
    _task,
    describe,
)

# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def board_cell_links(plan: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """The plan's inverse link: ticket id -> the plan rows whose board cell names it.

    A plan row's ``board`` cell is authoritative for the row it is tracked by, so
    it names the board task from the plan side. One ticket can be named by more
    than one row (a ticket whose ``done_when`` also holds untracked rows, or two
    rows that share one ticket), so every row is kept in document order: the
    first is the task's detail row and all of them are mirrored.
    """
    links: dict[str, list[dict[str, Any]]] = {}
    for row in (plan or {}).get("rows") or []:
        for ticket_id in row.get("board_ids") or []:
            rows = links.setdefault(ticket_id, [])
            if not any(existing["id"] == row["id"] for existing in rows):
                rows.append(row)
    return links


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

    links = board_cell_links(plan)
    for task in board_rows:
        # The plan's own board cell naming this task is the first link; a plan id
        # the ticket names, then the slug, follow. Every row naming the task is
        # mirrored; the first one is the detail row.
        linked_rows = links.get(task["id"]) or []
        matched = linked_rows[0] if linked_rows else None
        if matched is None:
            for plan_id in task["plan_ids"]:
                candidate = next((r for r in plan_rows if r["id"] == plan_id), None)
                if candidate is not None:
                    matched = candidate
                    break
        if matched is None:
            matched = match_by_slug(task["name"])
        if matched is not None:
            for row in linked_rows:
                matched_plan_ids.add(row["id"])
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

    links = board_cell_links(plan)
    for task in scope:
        # The plan is authoritative for a row it names: when a plan row's board
        # cell names this task, that row's gate and source paths are the task's,
        # and the cell is the declaration the dispatch is traced from.
        linked_rows = links.get(task["id"]) or []
        if not task["paths"] and not any(row["paths"] for row in linked_rows):
            findings.append(
                _finding(
                    RULE_NO_PATHS,
                    f"task {task['id']} ({task['name']}) names no source path",
                    task=task["id"],
                )
            )
        if not task["gates"] and not any(row["gates"] for row in linked_rows):
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
        if not kind and linked_rows:
            # A plan row's board cell names this task: the plan side carries the
            # declaration, so the task text need not repeat it.
            kind, declared = "linked", task["id"]
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
