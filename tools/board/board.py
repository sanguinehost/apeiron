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
             Prints the ``In progress`` predicate as a count (advisory).
``post``     write the reconciliation record as a project check-in, read-confirmed.
``set-status``  move one task's status, read-confirmed. Dry-run without ``--apply``.
``upsert``   the only creation path: resolve a normalized **creation key** against
             the whole board and create the row only if absent. The claim is taken
             before the board is read, so this is a lookup-or-create and not a
             check followed by a create. ``--apply`` to write; dry-run otherwise.
``lease``    the ``In progress`` predicate: a row is supported only by a live lane
             lease or a cycle-stamped artifact change. Advisory; ``--strict`` arms it.

Creation key
============

Identity is the key, never the title: a row's id is immutable and its name is a
display field. The key is ``--key``, else the first ``UPPER-hyphenated`` token in
the name, else the normalized name; a created row carries it as ``Key:`` in its
description, so a rename does not orphan it and a re-run resolves to it. Keys are
scoped per project. ``upsert --live`` resolves the key and refuses the create leg:
no create endpoint is verified on this instance, so the create runs through MCP
``operately.create_task`` with the claim taken by ``upsert`` first.

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
    python3 tools/board/board.py upsert --source board.json --key APEIRON-BOARD-W1-T9 \
        --name 'Board driver wave 1 task 9' --description-file row.md --apply
    python3 tools/board/board.py lease --live --project apeiron
    python3 tools/board/board.py --rules
    python3 -m unittest discover -s tools/board -p 'test_*.py'

Exit status: ``0`` success; ``1`` a finding (rule or drift); ``2`` usage,
transport or control-suite failure; ``3`` ``next`` found no actionable task.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from board_core import (  # noqa: F401 - re-exported: the entry's public surface
    ALL_RULES,
    BACKTICK,
    BOARD_DECLARATION,
    CONTRACT_RULES,
    DEFAULT_BASE_URL,
    DEFAULT_PROJECTS,
    DEFAULT_SPACE_ID,
    DIGEST,
    DUE_FIELD,
    EXTENSION_RULES,
    Finding,
    GATE_HEADS,
    HTTP_TIMEOUT_S,
    INTERNAL_MARKERS,
    ISO_DATE,
    KEY_ENV,
    LABEL_GATE,
    LABEL_PATHS,
    LABEL_RECEIPT,
    MIN_SCRUB_LENGTH,
    NO_ROW,
    PATH_EXTENSIONS,
    PROJECT_MENTION,
    PROJECT_PLANS,
    PROTOCOL_RULES,
    RECEIPT_PREFIXES,
    RECONCILE_RULES,
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
    RULE_PROJECT_MISSING,
    RULE_UNATTRIBUTED,
    RULE_UNKNOWN_MILESTONE,
    RULE_WRITE_NOT_LANDED,
    SCHEMA,
    TransportError,
    URL,
    _dedupe,
    _finding,
    _is_gate,
    _is_path,
    _labelled,
    _prosemirror,
    _split,
    _tokens,
    board_declaration,
    dated_facts,
    declared_row,
    due_date_of,
    flatten_description,
    gates_of,
    internal_named_by,
    paths_of,
    project_mentions,
    receipts_of,
    rules_of,
    slug,
)
from board_model import (  # noqa: F401 - re-exported: the entry's public surface
    ACTIONABLE,
    CANCELED_VALUES,
    CLOSED_MARKERS,
    DONE_VALUES,
    MIN_SLUG_MATCH,
    PLAN_ID_IN_CODE,
    TASK_COLUMNS,
    TASK_TABLE_KEYWORDS,
    _markdown_rows,
    _read_table,
    _require,
    _row_closed,
    attribute_projects,
    load_plan,
    next_task,
    non_closed_tasks,
    normalize_document,
    normalize_task,
    parse_markdown_plan,
    parse_plan_source,
    parse_tsv_plan,
    plan_from_document,
    select_tasks,
    sha256_of,
    sort_tasks,
    status_bucket,
)
from board_rest import (  # noqa: F401 - re-exported: the entry's public surface
    LIVE_COVERAGE_NOTES,
    RestClient,
    fetch_live,
)
from board_fixtures import (  # noqa: F401 - re-exported: the entry's public surface
    APEIRON,
    FIXTURE_BASELINE,
    FIXTURE_LEDGER_TSV,
    FIXTURE_PLAN_MD,
    SONG,
    _task,
    baseline_document,
    describe,
)
from board_check import (  # noqa: F401 - re-exported: the entry's public surface
    CONTRACT_SEEDED_DEFECTS,
    SEEDED_DEFECTS,
    _cell,
    _find,
    _fixture_check,
    _truncate,
    check,
    defect_board_declaration_missing,
    defect_board_declaration_names_another_row,
    defect_board_open_against_closed_plan,
    defect_board_row_missing_from_plan,
    defect_done_without_receipt,
    defect_due_date_without_fact,
    defect_milestone_without_tasks,
    defect_plan_row_missing_on_board,
    defect_task_without_gate,
    defect_write_did_not_land,
    document_plan,
    existing_paths,
    exit_code_for,
    full_check,
    plan_header,
    post_offline,
    project_name_hint,
    reconcile,
    reconciliation_record,
    render_check,
    render_next,
    render_table,
    resolve_project,
    verify_status_offline,
)
from board_controls import (  # noqa: F401 - re-exported: the entry's public surface
    CREATION_CONTROLS,
    run_controls,
)
from board_rows import (  # noqa: F401 - re-exported: the entry's public surface
    CREATION_RULES,
    RULE_CLOSED_ROW_IMMUTABLE,
    RULE_CREATE_UNKNOWN,
    RULE_DUPLICATE_KEY,
    RULE_KEY_CLAIM_HELD,
    ClaimStore,
    FileBoardTransport,
    RestTransport,
    cleanup,
    creation_key,
    default_state_dir,
    lease_audit,
    lookup_or_create,
    render_lease_audit,
    render_outcome,
    row_identities,
    row_key,
    scope_of,
    scratch_dir,
)


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


def run_upsert(args: argparse.Namespace) -> int:
    """The only creation path: resolve a key against the whole board, create it once.

    The claim is taken before the board is read (see ``board_rows``), so this is a
    lookup-or-create and not a check followed by a create.
    """
    if not args.source and not args.live:
        raise TransportError("upsert needs --source PATH (a JSON board) or --live")
    name = args.name or args.key
    if not name:
        raise TransportError("upsert needs --name NAME or --key KEY")
    description = args.description or ""
    if args.description_file:
        try:
            with open(args.description_file, encoding="utf-8") as handle:
                description = handle.read()
        except OSError as exc:
            raise TransportError(f"description file: {exc}") from None
    store = ClaimStore(args.state)
    if args.live:
        transport = RestTransport(space_id=args.space_id)
    else:
        transport = FileBoardTransport(args.source)
    outcome = lookup_or_create(
        transport,
        store,
        project=args.project or "",
        name=name,
        description=description,
        explicit_key=args.key or "",
        milestone_id=args.milestone,
        apply=args.apply,
        successes=args.successes or (),
        lease_s=args.lease_s,
        holder=args.holder or f"board.py:{os.getpid()}",
    )
    print(render_outcome(outcome))
    return 1 if outcome["action"] in ("duplicate", "refused") else 0


def run_lease(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    """The ``In progress`` predicate: a live lane lease or a cycle-stamped change.

    Advisory by default (exit 0), because arming it on the live board would stamp
    every existing row at once; ``--strict`` reddens on an unsupported row.
    """
    store = ClaimStore(args.state) if args.state else None
    rows = lease_audit(doc, store, repo_root=args.repo_root)
    print(render_lease_audit(rows, advisory=not args.strict))
    if args.strict:
        return 1 if any(not row["supported"] for row in rows) else 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="board.py", description="Operately board driver for the Sanguine stream."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="list",
        choices=("list", "next", "check", "post", "set-status", "upsert", "lease"),
        help="list | next | check | post | set-status | upsert | lease",
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
    # upsert: the creation key and the claim store
    parser.add_argument("--key", metavar="KEY", help="creation key (normalized identity token)")
    parser.add_argument("--name", metavar="NAME", help="row title; a display field, never identity")
    parser.add_argument("--description", metavar="TEXT", help="row description for a created row")
    parser.add_argument("--description-file", metavar="PATH", help="read the description from a file")
    parser.add_argument("--milestone", metavar="ID", help="milestone id for a created row")
    parser.add_argument(
        "--successes",
        action="append",
        metavar="ID",
        help="closed row this row succeeds; required before a closed row's key is reused",
    )
    parser.add_argument("--state", metavar="DIR", help=f"claim/lease store (default {default_state_dir()})")
    parser.add_argument("--holder", metavar="TEXT", help="who holds the claim, for the audit trail")
    parser.add_argument("--lease-s", type=int, default=300, help="claim lease in seconds")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="lease: redden on an unsupported In progress row (advisory without it)",
    )
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
        print("creation rules (identity is the key; creation is lookup-or-create):")
        for rule in CREATION_RULES:
            print(f"  {rule}")
        print("exit status: 0 clean, 1 finding, 2 usage/transport/control, 3 no actionable task")
        return 0
    if args.controls:
        code = run_controls()
        if code:
            return code
        if not (args.check or args.mode == "check"):
            return 0
    if args.mode == "upsert":
        try:
            return run_upsert(args)
        except TransportError as exc:
            print(f"board.py: {exc}", file=sys.stderr)
            return 2
    try:
        doc = load_document(args)
    except TransportError as exc:
        print(f"board.py: {exc}", file=sys.stderr)
        return 2
    args.project = resolve_project(doc, args.project)
    if args.mode == "lease":
        return run_lease(args, doc)
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
        # The In-progress predicate is reported, not enforced: it is advisory
        # unless --state names a claim store and --strict arms it. A count only,
        # so the check's own findings stay the red.
        print(
            render_lease_audit(
                lease_audit(
                    doc, ClaimStore(args.state) if args.state else None, repo_root=args.repo_root
                ),
                detail=False,
            )
        )
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
