#!/usr/bin/env python3
"""lane_modes: the modes tools/hooks/lane.py dispatches to.

One function per mode the CLI names: ``staged`` (the pre-commit
half), ``message PATH`` (the commit-msg half), ``capture``, ``rows`` and
``link``. Part of ``tools/hooks/lane.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from lane_core import (
    BOARD_DIR,
    BOARD_IMPORT_ERROR,
    DECLARATION_HINT,
    FetchError,
    REPO_ROOT,
    RULE_DRIVER_MISSING,
    RULE_FETCH_UNSTABLE,
    RULE_GIT,
    RULE_MISSING,
    RULE_ROW_CLOSED,
    RULE_ROW_NOT_IN_COMMIT,
    RULE_SNAPSHOT_STALE,
    RULE_UNAVAILABLE,
    _source_namespace,
    board,
    check_declaration,
    check_no_row,
    code_paths,
    commit_links,
    find_declaration,
    git,
    lane_of,
    load_board_documents,
    milestone_of,
    now_iso,
    project_commit,
    row_carried_in_change,
    snapshot_stamp,
    staged_added_lines,
    staged_change_epoch,
    staged_paths,
)
from lane_fetch import (
    client_from_environment,
    enumerated_board,
)

# The board of record is the sibling ``../indra`` ticket tree (AGENTS.md §9.3).
# The ticket block is the machine-readable object; the description is built from
# the scope so the old driver's gate/path/receipt readers see the same text.
INDRA_BOARD = REPO_ROOT.parent / "indra" / "board"
INDRA_TICKETS = INDRA_BOARD / "tickets"
TICKET_BLOCK = re.compile(
    r"```yaml[ \t]+(ticket|project)[ \t]*\n(.*?)\n```", re.S
)
INDRA_STATUS_LABELS = {
    "open": "Not started",
    "in_progress": "In progress",
    "blocked": "Blocked",
    "done": "Done",
    "cancelled": "Canceled",
}


def indra_board_document() -> dict[str, Any] | None:
    """The sibling ticket files as a normalized board document, or ``None``.

    The commit hook used to resolve a declaration only against Operately.  The
    board of record moved to ``../indra/board/tickets``; reading the ticket
    blocks directly keeps the declaration rule local, needs no credential and
    cannot read a stale capture as authority.  A missing sibling tree returns
    ``None`` so the old live/snapshot routes remain the fallback.
    """
    if not INDRA_TICKETS.is_dir():
        return None
    try:
        import yaml
    except ImportError:
        return None
    tasks: list[dict[str, Any]] = []
    milestones: dict[str, dict[str, Any]] = {}
    projects: dict[str, dict[str, Any]] = {}
    for path in sorted(INDRA_TICKETS.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = TICKET_BLOCK.search(text)
        if match is None or match.group(1) != "ticket":
            continue
        try:
            spec = yaml.safe_load(match.group(2))
        except yaml.YAMLError:
            continue
        if not isinstance(spec, dict):
            continue
        ticket_id = str(spec.get("id") or path.stem)
        project = str(spec.get("project") or "")
        milestone = str(spec.get("milestone") or "")
        state = str(spec.get("state") or "open")
        title = str(spec.get("title") or "")
        scope = spec.get("scope") if isinstance(spec.get("scope"), dict) else {}
        change = str(scope.get("change") or "")
        done_when = scope.get("done_when") if isinstance(scope.get("done_when"), list) else []
        description = "\n".join([change, *(str(item) for item in done_when)])
        tasks.append(
            {
                "id": ticket_id,
                "name": f"{ticket_id}: {title}",
                "status": {
                    "value": state,
                    "label": INDRA_STATUS_LABELS.get(state, state),
                },
                "milestone": {"id": milestone, "title": milestone},
                "project": {"id": project, "name": project},
                "description": description,
            }
        )
        if project:
            projects[project] = {"id": project, "name": project}
        if milestone:
            milestones[milestone] = {
                "id": milestone,
                "title": milestone,
                "project": {"id": project, "name": project},
            }
    if not tasks:
        return None
    return board.normalize_document(
        {
            "schema": board.SCHEMA,
            "space": {"id": "sanguine", "name": "sanguine"},
            "projects": list(projects.values()),
            "milestones": list(milestones.values()),
            "tasks": tasks,
        }
    )


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
            f"{board.declared_row(value)!r} against; the local ticket source is "
            f"{INDRA_TICKETS}, and a fallback is OPERATELY_API_KEY or "
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
        local = indra_board_document()
        if local is not None:
            return local, (
                f"local indra board {INDRA_TICKETS}: {len(local['tasks'])} ticket(s); "
                "the ticket block is the source and the worklog is the write path"
            )
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
