# Row: `apeiron-board-write-shape-FLHWsvrO3iZPMhoOtZ5r3t` (APEIRON-BOARD-WRITE-SHAPE).
#!/usr/bin/env python3
"""board_rest: the external v1 REST transport.

``RestClient`` holds the credential and scrubs it from every
message; ``fetch_live`` assembles a normalized document from the live board and
records what the external API cannot see. Part of ``tools/board/board.py``.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from typing import Iterable
from board_core import (
    DEFAULT_BASE_URL,
    DEFAULT_PROJECTS,
    DEFAULT_SPACE_ID,
    HTTP_TIMEOUT_S,
    KEY_ENV,
    MIN_SCRUB_LENGTH,
    SCHEMA,
    TransportError,
)
from board_model import (
    attribute_projects,
    normalize_document,
)

# ---------------------------------------------------------------------------
# Rich content
# ---------------------------------------------------------------------------


def rich_content(text: str) -> str:
    """A ProseMirror document, one paragraph per line, as a JSON string.

    ``projects/create_check_in`` and ``comments/create`` take rich content; a
    plain string is rejected with HTTP 500 and does **not** land (measured on
    the live instance). O(lines) over the input.
    """
    content: list[dict[str, Any]] = []
    for line in text.split("\n"):
        if line:
            content.append({"type": "paragraph", "content": [{"type": "text", "text": line}]})
        else:
            content.append({"type": "paragraph"})
    return json.dumps({"type": "doc", "content": content})


def resolve_status(statuses: list[dict[str, Any]], wanted: str) -> dict[str, Any] | None:
    """Match a status object by value, label or id, case-insensitively."""
    lowered = wanted.strip().lower()
    for status in statuses:
        for field in ("value", "label", "id"):
            if str(status.get(field) or "").lower() == lowered:
                return status
    return None


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class RestClient:
    """Read/write the external v1 API. The key never leaves this object."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        key: str | None = None,
        timeout: int = HTTP_TIMEOUT_S,
        opener: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self._key = key if key is not None else os.environ.get(KEY_ENV, "")
        if not self._key:
            raise TransportError(f"{KEY_ENV} is not set in the environment")
        self.timeout = timeout
        self._opener = opener or urllib.request.build_opener()

    def scrub(self, text: str) -> str:
        """Remove the credential from a message.

        A short value is not a credential and matching it would corrupt the text it
        appears in (a one-character stub key turns every ``k`` in a URL into a
        redaction), so an exact replacement is made only for a value long enough
        to be one. The ``Bearer`` form is always removed.
        """
        if self._key and len(self._key) >= MIN_SCRUB_LENGTH and self._key in text:
            text = text.replace(self._key, "<redacted>")
        return re.sub(r"(?i)(bearer\s+)\S+", r"\1<redacted>", text)

    def call(self, path: str, method: str = "GET", **params: Any) -> dict[str, Any]:
        params = {k: v for k, v in params.items() if v is not None}
        url = self.base_url + path
        if params and method == "GET":
            url += "?" + urllib.parse.urlencode(params)
        data = None
        headers = {"Authorization": "Bearer " + self._key}
        if method != "GET":
            data = json.dumps(params).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as exc:
            detail = self.scrub(exc.read().decode(errors="replace")[:200])
            raise TransportError(self.scrub(f"{path}: HTTP {exc.code}: {detail}")) from None
        except Exception as exc:  # noqa: BLE001 - surfaced scrubbed
            raise TransportError(self.scrub(f"{path}: {type(exc).__name__}: {exc}")) from None
        try:
            payload = json.loads(body)
        except ValueError:
            raise TransportError(self.scrub(f"{path}: response is not JSON")) from None
        if not isinstance(payload, dict):
            raise TransportError(f"{path}: response is not an object")
        return payload

    # -- verified reads ----------------------------------------------------

    def space_tasks(self, space_id: str) -> list[dict[str, Any]]:
        return list(self.call("spaces/list_tasks", space_id=space_id).get("tasks") or [])

    def projects(self) -> list[dict[str, Any]]:
        return list(self.call("projects/list").get("projects") or [])

    def milestones(self, project_id: str) -> list[dict[str, Any]]:
        return list(
            self.call("projects/list_milestones", project_id=project_id).get("milestones") or []
        )

    def milestone_tasks(self, milestone_id: str) -> list[dict[str, Any]]:
        return list(
            self.call("projects/list_milestone_tasks", milestone_id=milestone_id).get("tasks") or []
        )

    def task(self, task_id: str) -> dict[str, Any]:
        payload = self.call("tasks/get", id=task_id)
        task = payload.get("task")
        if not isinstance(task, dict):
            raise TransportError(f"tasks/get: no task for {task_id}")
        return task

    def check_ins(self, project_id: str) -> list[dict[str, Any]]:
        payload = self.call("projects/list_check_ins", project_id=project_id)
        for key in ("project_check_ins", "check_ins"):
            if isinstance(payload.get(key), list):
                return list(payload[key])
        return []

    # -- writer ------------------------------------------------------------

    def task_statuses(self, task_id: str) -> list[dict[str, Any]]:
        payload = self.call("tasks/list_task_statuses", task_id=task_id)
        return list(payload.get("task_statuses") or [])

    def update_status(self, task_id: str, status: str) -> str:
        """Apply a status. Returns ``landed`` or ``refused``.

        ``tasks/update_status`` takes the STATUS OBJECT, not its value: this
        instance answers ``400 Unknown input field`` for ``value=<str>`` and a
        silently non-landing ``500`` for ``status=<str>``. The object comes from
        ``tasks/list_task_statuses``, which also requires the task's ``type``.
        This instance answers a 500 for a mutation that landed, so a transport
        error is *not* evidence of refusal: the caller must read back. This method
        never decides; ``verify_status`` does.
        """
        try:
            type_ = (self.task(task_id).get("type")) or ""
            target = resolve_status(self.task_statuses(task_id), status)
            if not type_ or target is None:
                return "refused"
            self.call(
                "tasks/update_status",
                method="POST",
                task_id=task_id,
                type=type_,
                status=target,
            )
            return "landed"
        except TransportError:
            return "refused"

    def post_check_in(self, project_id: str, status: str, description: str) -> tuple[str, str]:
        """POST a check-in with rich content. Returns ``(transport, error)``.

        A plain-string description is rejected with HTTP 500 and does NOT land
        (measured on the live instance), so ``rich_content`` encodes it here.
        The transport is *not* the verdict — this instance 500s on a mutation
        that landed; the caller reads the check-in back.
        """
        try:
            self.call(
                "projects/create_check_in",
                method="POST",
                project_id=project_id,
                status=status,
                description=rich_content(description),
            )
            return "landed", ""
        except TransportError as exc:
            return "reported-error", str(exc)

    def create_check_in(self, project_id: str, status: str, description: str) -> str:
        transport, _ = self.post_check_in(project_id, status, description)
        return "landed" if transport == "landed" else "refused"

    def verify_status(self, task_id: str, target: str) -> tuple[bool, str]:
        observed = ((self.task(task_id).get("status") or {}).get("value")) or ""
        return observed == target, observed


LIVE_COVERAGE_NOTES = (
    "`projects/list_tasks` is not exposed over REST; project tasks are enumerated "
    "per milestone, so a project task carrying no milestone is not visible to --live",
    "task comments are not read over the external v1 API, so a receipt held only "
    "in a comment is not seen by --live",
    "`projects/list` is company-wide; --project filters by name over that inventory",
)


def fetch_live(
    client: RestClient,
    space_id: str = DEFAULT_SPACE_ID,
    projects: Iterable[str] = DEFAULT_PROJECTS,
) -> dict[str, Any]:
    """Assemble a normalized board document from the live board."""
    wanted = [p.lower() for p in projects]
    chosen: dict[str, dict[str, Any]] = {}
    for project in client.projects():
        name = (project.get("name") or "").lower()
        if not wanted or name in wanted:
            chosen[project["id"]] = {"id": project["id"], "name": project.get("name") or project["id"]}
    raw_tasks: dict[str, dict[str, Any]] = {}
    for task in client.space_tasks(space_id):
        raw_tasks[task["id"]] = task
    raw_milestones: list[dict[str, Any]] = []
    for project_id, project in chosen.items():
        for milestone in client.milestones(project_id):
            raw_milestones.append(
                {
                    "id": milestone.get("id"),
                    "title": milestone.get("title"),
                    "status": milestone.get("status"),
                    "project": project,
                }
            )
            for task in client.milestone_tasks(milestone["id"]):
                task.setdefault("milestone", {"id": milestone.get("id")})
                task.setdefault("project", project)
                raw_tasks.setdefault(task["id"], task)
    document = {
        "schema": SCHEMA,
        "origin": "live:operately",
        "space": {"id": space_id, "name": "sanguine"},
        "projects": list(chosen.values()),
        "milestones": raw_milestones,
        "tasks": list(raw_tasks.values()),
        "notes": list(LIVE_COVERAGE_NOTES),
    }
    normalized = normalize_document(document)
    attribute_projects(normalized)
    return normalized
