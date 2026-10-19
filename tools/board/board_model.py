#!/usr/bin/env python3
"""board_model: the sources of truth, the document, selection.

Plan parsing (markdown tables and the TSV ledgers), the
normalized schema-1 document every mode reads, and the selection and ordering
those modes print. Part of ``tools/board/board.py``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
from typing import Any
from typing import Iterable
from board_core import (
    SCHEMA,
    TransportError,
    _dedupe,
    _is_path,
    flatten_description,
    gates_of,
    internal_named_by,
    paths_of,
    project_mentions,
    receipts_of,
    slug,
)

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
    """Free row wording (a table with no state column): a marker anywhere closes it."""
    lowered = text.lower()
    return any(re.search(rf"\b{marker}\b", lowered) for marker in CLOSED_MARKERS)


# A state cell names its status first (`landed: <rev>`, `done; …`) or records the board move
# (`… board row Done`). A marker elsewhere is prose about the work, not the row's status:
# `open; … axioms closed …` is open.
_STATE_LEAD = re.compile(r"^[\s*_`~>\[(]*([a-z]+)")
_STATE_BOARD_DONE = re.compile(r"\bboard row (?:done|closed)\b")


def _state_closed(text: str) -> bool:
    lowered = text.lower()
    lead = _STATE_LEAD.match(lowered)
    if lead and lead.group(1) in CLOSED_MARKERS:
        return True
    return bool(_STATE_BOARD_DONE.search(lowered))


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
    # The board cell is the plan's inverse link: it names the board task the row
    # is tracked by. It is carried on the row so reconciliation can match a task
    # by its own id when the ticket does not name a plan row.
    board_at = column("board", "ticket")
    gate_at = column("gate", "gates")
    files_at = column("files", "file", "paths", "owned")
    state_at = column("state", "status")
    section_slug = slug(section) or "plan"
    rows: list[dict[str, Any]] = []
    for cells in body:
        row_text = " ".join(cells)
        if not row_text.strip(" |"):
            continue
        task_cell = cells[task_at] if task_at < len(cells) else ""
        # Aliases come from the id cell only: an id backticked elsewhere in the row is a mention
        # of another row, and as an alias it inherited this row's state.
        ids = _dedupe(PLAN_ID_IN_CODE.findall(task_cell)) or _dedupe(
            PLAN_ID_IN_CODE.findall(row_text)
        )[:1]
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
        if state_at is not None:
            # A `|` inside a cell (`|D|`) splits it; the last header column is still the last cell.
            state_cell = cells[-1] if state_at == len(header) - 1 else cells[state_at]
            if _state_closed(state_cell):
                closed_by = state_cell.strip()
        elif _row_closed(row_text):
            closed_by = "row wording"
        gate_text = cells[gate_at] if gate_at is not None and gate_at < len(cells) else ""
        files_text = cells[files_at] if files_at is not None and files_at < len(cells) else ""
        board_text = cells[board_at] if board_at is not None and board_at < len(cells) else ""
        board_ids = _dedupe(PLAN_ID_IN_CODE.findall(board_text))
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
                    "board_ids": board_ids,
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
                "closed": _state_closed(state),
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


# ---------------------------------------------------------------------------
# The ticket tree: the gate's own source of truth
# ---------------------------------------------------------------------------

# The indra ticket block. The commit hook projects these through
# `tools/hooks/lane_modes.py`; the gate reads the block directly, so the gate
# does not depend on the hook's normalized shape.
TICKET_YAML = re.compile(r"```yaml[ \t]+ticket[ \t]*\n(.*?)\n```", re.S)


def _ticket_description(spec: dict[str, Any]) -> str:
    """The task text: scope, gate, control, evidence and the board declaration.

    The contract rules read the description for the gate, the source paths, the
    receipt and the ``board:`` field. The ticket already carries each of those in
    its YAML, so the document is built from them rather than from a projection
    that drops them.
    """
    scope = spec.get("scope") if isinstance(spec.get("scope"), dict) else {}
    gate = spec.get("gate") if isinstance(spec.get("gate"), dict) else {}
    control = spec.get("control") if isinstance(spec.get("control"), dict) else {}
    evidence = spec.get("evidence") if isinstance(spec.get("evidence"), dict) else {}
    lines = [str(scope.get("change") or "")]
    lines.extend(str(item) for item in (scope.get("done_when") or []))
    paths = [str(path) for path in (scope.get("paths") or []) if str(path).strip()]
    if paths:
        lines.append("Source: " + ", ".join(f"`{path}`" for path in paths))
    command = str(gate.get("command") or "").strip()
    if command:
        cwd = str(gate.get("cwd") or ".")
        budget = gate.get("budget_s")
        suffix = f" (cwd {cwd}" + (f", {budget}s" if budget is not None else "") + ")"
        lines.append(f"Gate: `{command}`{suffix}")
    if control:
        lines.append(
            "Control: "
            f"{control.get('kind') or ''} -- {control.get('command') or ''}; "
            f"proves: {control.get('proves') or ''}"
        )
    grade = str(evidence.get("grade") or "").strip()
    if grade:
        lines.append(f"Grade: {grade}")
    receipt = evidence.get("receipt")
    if receipt:
        # `receipts_of` recognises a receipt path or a backticked span starting
        # with a receipt prefix. A prose receipt keeps its text behind the
        # prefix; a path is carried unchanged.
        value = str(receipt).replace("`", "'").strip()
        if _is_path(value):
            lines.append(f"Receipt: `{value}`")
        else:
            lines.append(f"Receipt: `receipt: {value}`")
    links = [str(link) for link in (spec.get("links") or []) if str(link).strip()]
    if links:
        lines.append("Link: " + ", ".join(f"`{link}`" for link in links))
    lines.append(f"Board: `{spec.get('id')}`")
    return "\n".join(line for line in lines if line.strip())


def tickets_document(directory: str) -> dict[str, Any]:
    """Build a raw schema-1 document from ``<dir>/*.md`` indra ticket files."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - the hook path already requires it
        raise TransportError("tickets: PyYAML is not installed") from None
    try:
        names = sorted(name for name in os.listdir(directory) if name.endswith(".md"))
    except OSError as exc:
        raise TransportError(f"tickets: {exc}") from None
    tasks: list[dict[str, Any]] = []
    projects: dict[str, dict[str, Any]] = {}
    milestones: dict[str, dict[str, Any]] = {}
    for name in names:
        path = os.path.join(directory, name)
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise TransportError(f"tickets: {exc}") from None
        match = TICKET_YAML.search(text)
        if match is None:
            continue
        try:
            spec = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            raise TransportError(f"tickets: {path}: {exc}") from None
        if not isinstance(spec, dict):
            continue
        ticket_id = str(spec.get("id") or os.path.splitext(name)[0])
        project = str(spec.get("project") or "")
        milestone = str(spec.get("milestone") or "")
        state = str(spec.get("state") or "open")
        tasks.append(
            {
                "id": ticket_id,
                "name": f"{ticket_id}: {spec.get('title') or ''}",
                "status": {"value": state, "label": state},
                "project": {"id": project, "name": project} if project else None,
                "milestone": {"id": milestone, "title": milestone} if milestone else None,
                "description": _ticket_description(spec),
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
        raise TransportError(f"tickets: no ticket block under {directory}")
    return {
        "schema": SCHEMA,
        "space": {"id": "sanguine", "name": "sanguine"},
        "projects": list(projects.values()),
        "milestones": list(milestones.values()),
        "tasks": tasks,
    }


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
