#!/usr/bin/env python3
# Row: `apeiron-harness-board-for-tuBBxdMJdF2fUG7M8Oygm` (APEIRON-HARNESS-BOARD-MUTATION).
"""mutate: read-back-verified Operately mutations for a lane shell.

The ``operately`` MCP namespace is not reachable from a lane's shell, so the
external v1 REST API (``board_rest.RestClient``, already used by ``board.py``)
is the lane-usable write path. This entry exposes the four mutations a lane
performs -- ``create-task``, ``update-status``, ``comment``, ``create-check-in``
-- and makes each one's verdict the *read-back*, never the mutation response.

Why the read-back is the verdict (three measured failure modes)
===============================================================

1. **The mutation lies about success.** Over MCP, ``create_task`` and
   ``update_task_status`` return
   ``"Structured content does not match the tool's output schema"`` while the
   mutation lands. A caller that treats that as failure retries and duplicates a
   row; one that ignores it may miss a real failure. Over REST the same shape
   appears as HTTP **500** with no state change reported: the request may or may
   not have landed and the status code does not say which.

2. **The mutation silently does not land.** ``tasks/update_status`` with a bare
   ``value``/``status``/``status_id`` string answers **400** (``value``) or
   **500 with no landing** (bare ``status``/``status_id``) because the endpoint
   takes a full ``task_status`` *object* (fields ``value``, ``label``, ``id``,
   ``color``, ``index``, ``closed``), not a string. ``create_check_in`` and
   ``comments/create`` likewise answer 500 and do **not** land unless
   ``description``/``content`` is a ProseMirror document encoded as a JSON
   string -- a plain string is silently rejected. A successful-looking call is
   therefore not evidence of a landed write.

3. **The absent MCP namespace.** A lane's runtime may have no ``operately`` MCP
   server at all (``search`` returns 0 items). This script needs only
   ``$OPERATELY_API_KEY`` (or ``.env`` in the repository root) and ``urllib``.

Shapes the instance was observed to accept and reject (measured on the live
apeiron board this cycle, base ``87d904f``)
--------------------------------------------------------------------------------

* ``tasks/update_status``: ``{"value": "in_progress"}`` -> HTTP 400
  ``Unknown input field: value``; ``{"status": "in_progress"}`` (a bare string)
  -> HTTP 500 with the row unchanged; ``{"status_id": "<uuid>"}`` -> HTTP 400
  ``Unknown input field: status_id``. The accepted shape is
  ``{"task_id": ..., "type": "project"|"space", "status": <the full object from
  tasks/list_task_statuses>}``, confirmed by read-back. (The dispatch that
  commissioned this tool reported ``status_id`` as 500-no-landing; the live
  instance answers 400 ``Unknown input field`` -- the measured value is used.)
* ``tasks/create``: requires ``type`` (``project``/``space``), ``id`` (the
  **project id** for ``type=project``, the **space id** for ``type=space`` --
  not a new task id), ``name`` and ``due_date``. ``due_date`` is required and
  nullable, so it must be sent as an explicit JSON ``null`` (omitting it is a
  400; ``{}`` is a 500). The new task id is assigned by the server and returned.
* ``comments/create`` and ``projects/create_check_in``: ``content``/
  ``description`` must be a ProseMirror document as a JSON string
  (``json.dumps({"type": "doc", ...})``). A plain string -- even one that looks
  like a comment -- answers 500 and does not land.

Trust boundary
==============

* The verdict is the read-back object read through ``GET`` endpoints
  (``tasks/get``, ``comments/list``, ``projects/list_check_ins``). Exit status
  is ``0`` only when the read-back shows the intended state; a mutation whose
  read-back disagrees exits ``1`` and prints the observed value.
* ``tasks/create``'s ``description``, ``comments/create``'s ``content`` and
  ``create_check_in``'s ``description`` are converted from plain text to a
  ProseMirror paragraph-per-line document here. The reader is
  ``board.flatten_description``.
* The credential never leaves ``RestClient``; a 500 response body is scrubbed
  by ``RestClient.scrub`` before it is printed. ``create-task`` needs one POST
  with an explicit JSON null, which ``RestClient.call`` cannot express (it drops
  ``None``); ``Client.post`` reuses the same opener, base URL, timeout and key
  and scrubs identically.
* This is not a proof that a *creation* is unique -- ``board.py upsert`` owns the
  lookup-or-create claim. A lane that must not duplicate a row still resolves
  the row first.
* An MCP caller that can reach the ``operately`` server remains the path the
  board tooling documents; this script is the shell fallback and neither
  replaces nor weakens that.

Usage
=====

::

    python3 tools/board/mutate.py create-task --project <project_id> \
        --name APEIRON-X --milestone <milestone_id> --description-file row.md
    python3 tools/board/mutate.py update-status <task_id> in_progress
    python3 tools/board/mutate.py comment <task_id> --content "lane return"
    python3 tools/board/mutate.py create-check-in --project <project_id> \
        --status caution --content-file note.md

Exit status: ``0`` the read-back confirms the intended state; ``1`` the read-back
disagrees or the target state is not available (the observed state is printed);
``2`` usage or transport failure before any read-back could be made.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import board  # noqa: E402 -- the sibling driver; its REST client is reused verbatim

EXIT_OK = 0
EXIT_NOT_LANDED = 1
EXIT_USAGE = 2
RULE_NOT_LANDED = board.RULE_WRITE_NOT_LANDED

# The endpoint's own enum, measured from its HTTP 400
# `Invalid value for enum project_check_in_status: red. Allowed values:
# on_track, caution, off_track`.
CHECK_IN_STATUSES = ("on_track", "caution", "off_track")


class MutateError(RuntimeError):
    """A usage or transport failure that prevents a read-back."""


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_env(root: str) -> None:
    """Put ``.env``'s ``OPERATELY_API_KEY`` in the environment when unset.

    The token is untracked and owner-provided (AGENTS.md §9.3). The value is
    handed to ``RestClient`` and never printed.
    """
    if os.environ.get(board.KEY_ENV):
        return
    env_path = os.path.join(root, ".env")
    try:
        with open(env_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() == board.KEY_ENV:
                    os.environ[board.KEY_ENV] = value.strip().strip('"').strip("'")
                    return
    except OSError:
        return


def rich_content(text: str) -> str:
    """A ProseMirror document, one paragraph per line, as a JSON string.

    The four write endpoints take rich content; a plain string is rejected with
    HTTP 500 and does not land. O(lines) over the input.
    """
    content: list[dict[str, Any]] = []
    for line in text.split("\n"):
        if line:
            content.append(
                {"type": "paragraph", "content": [{"type": "text", "text": line}]}
            )
        else:
            content.append({"type": "paragraph"})
    return json.dumps({"type": "doc", "content": content})


def stamp_of(text: str) -> str:
    """The first non-empty line of a text block: the read-back stamp."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def read_text(
    args: argparse.Namespace, file_attr: str = "content_file", text_attr: str = "content"
) -> str:
    path = getattr(args, file_attr, None)
    if path:
        try:
            with open(path, encoding="utf-8") as handle:
                return handle.read()
        except OSError as exc:
            raise MutateError(f"content file: {exc}") from None
    return getattr(args, text_attr, "") or ""


class Client(board.RestClient):
    """``RestClient`` plus a POST that can send an explicit JSON null.

    ``tasks/create`` requires ``due_date`` present-and-null; ``RestClient.call``
    drops ``None`` values, so ``due_date`` would be omitted and the create
    refused 400. ``post`` reuses the same opener, key, timeout and scrubber.
    """

    def post(self, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
        url = self.base_url + path
        data = json.dumps(payload).encode()
        headers = {
            "Authorization": "Bearer " + self._key,
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                code = response.status
                body = response.read().decode()
        except urllib.error.HTTPError as exc:
            code = exc.code
            body = exc.read().decode(errors="replace")[:300]
        except Exception as exc:  # noqa: BLE001 - surfaced scrubbed
            raise board.TransportError(
                self.scrub(f"{path}: {type(exc).__name__}: {exc}")
            ) from None
        try:
            parsed: Any = json.loads(body)
        except ValueError:
            parsed = self.scrub(body)
        if isinstance(parsed, str):
            parsed = self.scrub(parsed)
        return code, parsed


def report_mutation(endpoint: str, code: int, response: Any) -> None:
    print(f"mutation {endpoint} transport=HTTP {code}")
    print(f"  mutation-response: {json.dumps(response, sort_keys=True)}")


def report_readback(observed: Any) -> None:
    print(f"  read-back: {json.dumps(observed, sort_keys=True)}")


def disagree(expected: str, observed: Any) -> int:
    print(
        f"  verdict=NOT-landed rule={RULE_NOT_LANDED} expected={expected!r} "
        f"observed={json.dumps(observed, sort_keys=True)}"
    )
    return EXIT_NOT_LANDED


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_create_task(args: argparse.Namespace, client: Client) -> int:
    text = read_text(args, "description_file", "description")
    payload: dict[str, Any] = {
        "type": args.type,
        "id": args.project,  # the project id (type=project) or space id (type=space)
        "name": args.name,
        "due_date": None,
    }
    if args.milestone:
        payload["milestone_id"] = args.milestone
    if text:
        payload["description"] = rich_content(text)
    code, response = client.post("tasks/create", payload)
    report_mutation("POST tasks/create", code, response)
    task_id = ""
    if isinstance(response, dict):
        task_id = str((response.get("task") or {}).get("id") or "")
    if code != 200 or not task_id:
        print(f"  verdict=NOT-landed rule={RULE_NOT_LANDED} no task id in the response")
        return EXIT_NOT_LANDED
    # `tasks/get` only preloads the milestone when asked; the query value is the
    # string `true` (Python `True` urlencodes to `True` and answers 400).
    observed = client.call("tasks/get", id=task_id, include_milestone="true").get("task") or {}
    report_readback(observed)
    milestone_ok = True
    if args.milestone:
        milestone_ok = ((observed.get("milestone") or {}).get("id")) == args.milestone
    if observed.get("name") == args.name and milestone_ok:
        print(f"  verdict=landed task={task_id}")
        return EXIT_OK
    print(f"  verdict=NOT-landed rule={RULE_NOT_LANDED} task={task_id}")
    return EXIT_NOT_LANDED


def resolve_status(statuses: list[dict[str, Any]], wanted: str) -> dict[str, Any] | None:
    """Match a status object by value, label or id, case-insensitively."""
    lowered = wanted.strip().lower()
    for status in statuses:
        for field in ("value", "label", "id"):
            if str(status.get(field) or "").lower() == lowered:
                return status
    return None


def cmd_update_status(args: argparse.Namespace, client: Client) -> int:
    task = client.task(args.task_id)
    type_ = args.type or task.get("type") or ""
    if type_ not in ("project", "space"):
        raise MutateError(
            f"tasks/get did not report a type for {args.task_id}; pass --type project|space"
        )
    payload = client.call("tasks/list_task_statuses", task_id=args.task_id)
    statuses = list(payload.get("task_statuses") or [])
    target = resolve_status(statuses, args.status)
    if target is None:
        available = [s.get("value") for s in statuses]
        print(
            f"update-status {args.task_id} target={args.status!r} is not an available "
            f"status; available values={available} labels="
            f"{[s.get('label') for s in statuses]}"
        )
        return EXIT_NOT_LANDED
    code, response = client.post(
        "tasks/update_status",
        {"task_id": args.task_id, "type": type_, "status": target},
    )
    report_mutation("POST tasks/update_status", code, response)
    observed = client.task(args.task_id)
    report_readback(observed)
    observed_value = ((observed.get("status") or {}).get("value")) or ""
    if observed_value == target["value"]:
        print(f"  verdict=landed status={observed_value}")
        return EXIT_OK
    return disagree(target["value"], observed_value)


def cmd_comment(args: argparse.Namespace, client: Client) -> int:
    text = read_text(args)
    if not text.strip():
        raise MutateError("comment needs --content or --content-file")
    code, response = client.post(
        "comments/create",
        {
            "entity_id": args.entity_id,
            "entity_type": args.entity_type,
            "content": rich_content(text),
        },
    )
    report_mutation("POST comments/create", code, response)
    stamp = stamp_of(text)
    payload = client.call(
        "comments/list", entity_id=args.entity_id, entity_type=args.entity_type
    )
    comments = list(payload.get("comments") or [])
    match = None
    for comment in comments:
        if stamp in board.flatten_description(comment.get("content")):
            match = comment
            break
    report_readback(
        {
            "comments_read": len(comments),
            "matched": match is not None,
            "comment_id": (match or {}).get("id"),
        }
    )
    if match is not None:
        print(f"  verdict=landed comment={match.get('id')} entity={args.entity_id}")
        return EXIT_OK
    return disagree(stamp, {"comments_read": len(comments), "text": stamp})


def cmd_create_check_in(args: argparse.Namespace, client: Client) -> int:
    text = read_text(args)
    if not text.strip():
        raise MutateError("create-check-in needs --content or --content-file")
    if args.status not in CHECK_IN_STATUSES:
        raise MutateError(
            f"--status {args.status!r} is not in {', '.join(CHECK_IN_STATUSES)} "
            "(this endpoint's enum, measured from its HTTP 400)"
        )
    code, response = client.post(
        "projects/create_check_in",
        {
            "project_id": args.project,
            "status": args.status,
            "description": rich_content(text),
        },
    )
    report_mutation("POST projects/create_check_in", code, response)
    stamp = stamp_of(text)
    matches = [
        entry
        for entry in client.check_ins(args.project)
        if stamp in board.flatten_description(entry.get("description") or entry.get("content"))
    ]
    report_readback({"check_ins_matched": len(matches), "stamp": stamp})
    if matches:
        print(f"  verdict=landed check_in={matches[-1].get('id')}")
        return EXIT_OK
    return disagree(stamp, {"check_ins_matched": 0})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mutate.py",
        description="Read-back-verified Operately mutations for a lane shell.",
    )
    parser.add_argument("--base-url", default=board.DEFAULT_BASE_URL, help="external v1 API base")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-task", help="create a project or space task, read back")
    create.add_argument("--project", required=True, help="project id (type=project) or space id")
    create.add_argument("--name", required=True, help="task name")
    create.add_argument("--milestone", help="milestone id to attach")
    create.add_argument("--type", default="project", choices=("project", "space"))
    create.add_argument("--description", default="", help="task description (plain text)")
    create.add_argument("--description-file", help="read the description from a file")

    status = sub.add_parser("update-status", help="move a task's status, read back")
    status.add_argument("task_id")
    status.add_argument("status", help="target status value, label or id")
    status.add_argument("--type", choices=("project", "space"), help="override the task type")

    comment = sub.add_parser("comment", help="add a task comment, read back")
    comment.add_argument("entity_id")
    comment.add_argument("--entity-type", default="project_task")
    comment.add_argument("--content", default="")
    comment.add_argument("--content-file")

    checkin = sub.add_parser("create-check-in", help="post a project check-in, read back")
    checkin.add_argument("--project", required=True)
    checkin.add_argument("--status", required=True, choices=CHECK_IN_STATUSES)
    checkin.add_argument("--content", default="")
    checkin.add_argument("--content-file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env(repo_root())
    try:
        client = Client(base_url=args.base_url)
    except board.TransportError as exc:
        print(f"mutate.py: {exc}", file=sys.stderr)
        return EXIT_USAGE
    handler = {
        "create-task": cmd_create_task,
        "update-status": cmd_update_status,
        "comment": cmd_comment,
        "create-check-in": cmd_create_check_in,
    }[args.command]
    try:
        return handler(args, client)
    except (MutateError, board.TransportError) as exc:
        print(f"mutate.py: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
