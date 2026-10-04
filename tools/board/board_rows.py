#!/usr/bin/env python3
# Row: APEIRON-BOARD-GATEWAY (apeiron-board-gateway-or-EFhF7A8mPQSyWKW45mp0LO)
"""board_rows: the creation key, atomic lookup-or-create, the In-progress predicate.

Part of ``tools/board/board.py``. Three mechanisms, one contract each.

**Identity is the key, not the title.** A row's ``id`` is immutable and its name
is a display field; neither is ever used to decide whether a row already
exists. The **creation key** is the normalized identity token of the row:
``--key APEIRON-BOARD-W1-T2``, else the first ``UPPER-hyphenated`` token in the
name, else the normalized name. A row carries its key in its description as a
labelled ``Key:`` field, so re-running a creation after a rename still resolves
to the same row. Keys are scoped per project, so the same token in two projects
is two rows.

**Creation is an atomic lookup-or-create, not check-then-create.** Three
duplicate rows in one night were all the same shape: create first, discover the
twin after. The window is closed by claiming the key *before* the board is read:
a claim is a file created with ``O_CREAT|O_EXCL`` (atomic on POSIX), the board
read happens inside that critical section, the create happens inside it, and the
create is read back before the claim is released. A second creator that finds a
live claim is refused rather than allowed to race. A claim whose holder is dead
or whose lease has expired is taken over, and the takeover is recorded, so a
create killed between the write and the recording is recovered by the retry: the
retry re-reads the board *inside the claim*, finds the row the dead create left
behind, and returns it instead of creating a twin.

**``In progress`` is evidence-bearing.** A row in that state is *supported* only
if it carries a live lane lease (an unexpired claim naming this row's scope) or
a cycle-stamped artifact change (a ``Cycle: YYYY-MM-DD`` field plus a declared
source path whose mtime date is not older than that cycle). The predicate is
**advisory** in ``--check`` and in ``lease`` mode by default and exits 0; it
reddens only under ``--strict``, because arming it on the live board would stamp
every existing row at once and that is a lead decision, not a tool decision.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import shutil
import tempfile
import time
from datetime import date
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Iterable

from board_core import SCHEMA, TransportError, flatten_description
from board_model import attribute_projects, normalize_document, status_bucket

# ---------------------------------------------------------------------------
# Rule names (stable interface, as in board_core)
# ---------------------------------------------------------------------------

RULE_DUPLICATE_KEY = "duplicate-creation-key"
RULE_KEY_CLAIM_HELD = "creation-key-claim-held"
RULE_CLOSED_ROW_IMMUTABLE = "closed-row-needs-linked-successor"
RULE_CREATE_UNKNOWN = "create-outcome-unknown"
CREATION_RULES = (
    RULE_DUPLICATE_KEY,
    RULE_KEY_CLAIM_HELD,
    RULE_CLOSED_ROW_IMMUTABLE,
    RULE_CREATE_UNKNOWN,
)

# ---------------------------------------------------------------------------
# The creation key
# ---------------------------------------------------------------------------

KEY_MAX = 96
# The labelled field a created row carries, so identity survives a retitle.
KEY_FIELD = re.compile(r"(?:^|\n)[ \t>*_-]*key\s*[:：]\s*([^\n]+)", re.I)
# ``APEIRON-BOARD-W1-T2``, ``G1-REG-1``, ``W2-T1``: an identity token in a name.
IDENTITY_TOKEN = re.compile(r"\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\b")
# The cycle stamp a row carries to claim its artifact moved in this cycle.
CYCLE_FIELD = re.compile(r"(?:^|\n)[ \t>*_-]*cycle(?:\s+id)?\s*[:：]\s*([^\n]+)", re.I)
# A lease older than this, whose holder is gone, may be taken over.
LEASE_S = 300


def normalize_key(raw: str) -> str:
    """The normalized form of a creation key.

    Uppercase; every run of characters outside ``A-Z0-9`` becomes one ``-``;
    leading and trailing ``-`` are dropped and the result is truncated to
    ``KEY_MAX``. ``G1/REG-1``, ``g1 reg 1`` and ``G1-REG-1`` are one key.
    """
    folded = re.sub(r"[^A-Z0-9]+", "-", (raw or "").upper()).strip("-")
    return folded[:KEY_MAX].strip("-") if len(folded) > KEY_MAX else folded


def creation_key(name: str, explicit: str = "") -> str:
    """The creation key of a row about to be created.

    Precedence: the caller's ``--key``; else the first identity token in the
    name; else the whole name normalized. O(1): three bounded regex passes over
    a name of at most a few hundred characters.
    """
    if (explicit or "").strip():
        return normalize_key(explicit)
    # ``/`` and ``_`` are token separators in this programme's spelling
    # (``G1/REG-1``, ``W1_I01``), so they are folded before the search: a name
    # spelling two ways must yield one key.
    token = IDENTITY_TOKEN.search(re.sub(r"[/_]+", "-", name or ""))
    if token:
        return normalize_key(token.group(1))
    return normalize_key(name or "")


def declared_key(text: str) -> str:
    """The key a row's own ``Key:`` field declares, or ``""``."""
    match = KEY_FIELD.search(flatten_description(text))
    if not match:
        return ""
    first = re.sub(r"[`.,;:]+$", "", match.group(1).strip().split(" ", 1)[0])
    return normalize_key(first)


def row_key(task: dict[str, Any]) -> str:
    """A row's primary identity: its declared key, else the plan row it declares,
    else the key its name yields. One pass over at most three fields per row,
    independent of the row count."""
    declared = declared_key(task.get("text"))
    if declared:
        return declared
    for plan_id in task.get("plan_ids") or []:
        if normalize_key(plan_id):
            return normalize_key(plan_id)
    return creation_key(task.get("name") or "")


def row_identities(task: dict[str, Any]) -> set[str]:
    """Every key this row answers to: its declared ``Key:``, the plan row its
    ``board:`` field names, and the key its name yields.

    A row created before this path existed carries no ``Key:`` field, so the plan
    row it declares is its identity; that is why a lookup matches on the union
    rather than on one field. O(fields) per row, bounded regex work.
    """
    found = {declared_key(task.get("text")), creation_key(task.get("name") or "")}
    found.update(normalize_key(plan_id) for plan_id in task.get("plan_ids") or [])
    found.discard("")
    return found


def scope_of(project: str, key: str) -> str:
    """The claim scope: ``<project>/<key>``, project-scoped so keys never collide."""
    return f"{(project or 'space').strip().lower() or 'space'}/{normalize_key(key) or 'unnamed'}"


def scope_file(scope: str) -> str:
    """A scope as one filename. O(|scope|): one regex pass, characters outside the
    safe set collapse to ``-``; the result is unique per scope because a scope
    contains no other separator."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", scope) + ".json"


def successor_scope(scope: str, generation: int) -> str:
    """The scope of a linked successor of a closed row (never the closed row's own)."""
    return f"{scope}.S{generation}"


# ---------------------------------------------------------------------------
# The claim store: the serialization point, and the lane lease
# ---------------------------------------------------------------------------


class CreateOutcomeUnknown(RuntimeError):
    """The create call's outcome cannot be known: the row may or may not exist.

    A transport raises this instead of ``TransportError`` when the request was
    dispatched but no reply came back. The claim is then released *without*
    recording a row id, and the retry's in-claim board read decides.
    """


def pid_alive(pid: int | None) -> bool:
    """Whether a holder pid is still running. An unknown pid is treated as alive,
    so the lease expiry is the only escape from a claim nobody can prove is dead."""
    if pid is None:
        return True
    try:
        os.kill(int(pid), 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def default_state_dir() -> str:
    """Where claims live. Outside the repository: the claim is process state, and a
    tracked file would put a lock under ``git``. ``APEIRON_BOARD_STATE`` overrides."""
    override = os.environ.get("APEIRON_BOARD_STATE")
    if override:
        return override
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache, "apeiron", "board")


def scratch_dir(prefix: str = "board-") -> str:
    """A disposable directory. Prefers ``/tmp/opencode`` so control copies do not
    consume the ``/tmp`` tmpfs (``AGENTS.md`` §10, tmpfs exhaustion row)."""
    root = os.environ.get("APEIRON_BOARD_TMP") or "/tmp/opencode"
    parent = root if os.path.isdir(root) else tempfile.gettempdir()
    return tempfile.mkdtemp(prefix=prefix, dir=parent)


class ClaimStore:
    """One directory, one ``O_EXCL`` claim file per scope.

    ``acquire`` is the only operation that creates state and it is atomic:
    ``os.open(O_CREAT|O_EXCL)`` succeeds for exactly one caller. Every other
    operation reads or rewrites a claim this process already owns.
    """

    def __init__(self, root: str | None = None) -> None:
        self.root = root or default_state_dir()
        self.dir = os.path.join(self.root, "claims")

    def path_for(self, scope: str) -> str:
        return os.path.join(self.dir, scope_file(scope))

    def read(self, scope: str) -> dict[str, Any]:
        """The claim on a scope, or ``{}``. One ``open`` and one ``json.load`` per call,
        1 file touched and 0 writes; a corrupt claim reads as present-but-unknown,
        which refuses a takeover rather than granting one."""
        try:
            with open(self.path_for(scope), encoding="utf-8") as handle:
                raw = json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            return {"scope": scope, "corrupt": True}
        return raw if isinstance(raw, dict) else {"scope": scope, "corrupt": True}

    def acquire(
        self,
        scope: str,
        holder: str = "",
        now: float | None = None,
        lease_s: int = LEASE_S,
    ) -> tuple[dict[str, Any] | None, str]:
        """Claim a scope. Returns ``(claim, "held")``, ``(None, "held-by-OTHER")``
        or ``(None, "unreadable")``.

        O(1) file operations: one ``O_EXCL`` create, or one read plus one atomic
        ``os.replace`` when the existing claim's holder is dead or its lease has
        expired.
        """
        now = time.time() if now is None else now
        os.makedirs(self.dir, exist_ok=True)
        path = self.path_for(scope)
        claim = {
            "scope": scope,
            "holder": holder or f"pid:{os.getpid()}",
            "pid": os.getpid(),
            "acquired_at": now,
            "lease_s": lease_s,
            "task_id": "",
            "takeover_of": "",
        }
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            existing = self.read(scope)
            if existing.get("corrupt"):
                return None, "unreadable"
            acquired = float(existing.get("acquired_at") or 0.0)
            expired = (now - acquired) > float(existing.get("lease_s") or LEASE_S)
            if not (expired or not pid_alive(existing.get("pid"))):
                return None, "held-by-OTHER"
            claim["takeover_of"] = f"{existing.get('holder')}@{acquired:.0f}"
            self._write(path, claim)
            return claim, "taken-over"
        except OSError:
            return None, "unreadable"
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(claim, stream)
        return claim, "held"

    def _write(self, path: str, claim: dict[str, Any]) -> None:
        """Replace a claim file atomically, so a reader never sees a partial one."""
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(claim, stream)
        os.replace(temporary, path)

    def record(self, claim: dict[str, Any], task_id: str) -> None:
        """Bind a created row id to a claim this process holds."""
        claim = dict(claim, task_id=task_id)
        self._write(self.path_for(claim["scope"]), claim)

    def release(self, claim: dict[str, Any]) -> None:
        """Drop a claim. O(1): one unlink, and a missing file is already released."""
        try:
            os.unlink(self.path_for(claim["scope"]))
        except FileNotFoundError:
            pass

    def lease(self, scope: str, now: float | None = None) -> dict[str, Any]:
        """The lane lease on a scope, with its liveness. One ``open`` and one
        ``json.load`` per call, 1 file touched and 0 writes.

        ``live`` means a lane is working this scope right now: the claim is
        unexpired *and* its holder is a running process. A released claim is not a
        lease (the work is over) and neither is one whose holder has exited.
        Expiry governs takeover in ``acquire``; here it also revokes the lease.
        """
        now = time.time() if now is None else now
        info = self.read(scope)
        if not info or info.get("corrupt"):
            return {"scope": scope, "present": bool(info), "live": False}
        acquired = float(info.get("acquired_at") or 0.0)
        lease_s = float(info.get("lease_s") or LEASE_S)
        expired = (now - acquired) > lease_s
        alive = pid_alive(info.get("pid"))
        live = (not expired) and alive
        return {
            "scope": scope,
            "present": True,
            "live": bool(live),
            "expired": expired,
            "holder_alive": alive,
            "holder": info.get("holder") or "",
            "pid": info.get("pid"),
            "task_id": info.get("task_id") or "",
            "age_s": round(now - acquired, 3),
            "lease_s": lease_s,
            "takeover_of": info.get("takeover_of") or "",
        }


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class FileBoardTransport:
    """A board held in a JSON file: the hermetic leg, and what the controls drive.

    ``board()`` is O(rows) over the document's tasks (a full normalize, by
    definition the whole board). ``create_task`` is O(|document|) for the
    read-modify-write and takes an exclusive ``flock`` on a sidecar so two
    processes cannot interleave their rewrites; ``os.replace`` makes the rewrite
    itself atomic.
    """

    def __init__(self, path: str, id_factory=None) -> None:
        self.path = path
        self._id_factory = id_factory or (lambda: os.urandom(4).hex())

    def _raw(self) -> dict[str, Any]:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def board(self) -> dict[str, Any]:
        doc = normalize_document(self._raw())
        attribute_projects(doc)
        return doc

    def create_task(
        self,
        project_id: str,
        name: str,
        description: str,
        milestone_id: str | None = None,
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        lock_path = f"{self.path}.lock"
        with open(lock_path, "w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            raw = self._raw()
            task_id = f"{creation_key(name)[:40].lower()}-{self._id_factory()}"
            raw.setdefault("tasks", []).append(
                {
                    "id": task_id,
                    "name": name,
                    "status": {"value": "pending", "label": "Not started"},
                    "description": description,
                    "project": {"id": project_id, "name": project_id},
                    "milestone": {"id": milestone_id} if milestone_id else None,
                }
            )
            temporary = f"{self.path}.{os.getpid()}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(raw, handle)
            os.replace(temporary, self.path)
        return task_id


class RestTransport:
    """The live board: the lookup leg only.

    ``create_task`` is refused. The external v1 API exposes no create endpoint
    this tool has verified on this instance (``tasks/update_status`` answers 500
    for a mutation that landed, and ``spaces/create_task`` is not in the verified
    set), so the create leg runs through the MCP ``operately.create_task`` tool
    with the claim taken here first. Refusing is fail-closed; guessing an
    endpoint and reporting a create that did not happen is not.
    """

    def __init__(self, client=None, space_id: str | None = None) -> None:
        from board_rest import DEFAULT_SPACE_ID, RestClient, fetch_live

        self.client = client or RestClient()
        self.space_id = space_id or DEFAULT_SPACE_ID

    def board(self) -> dict[str, Any]:
        from board_rest import fetch_live

        return fetch_live(self.client, self.space_id)

    def create_task(self, *args: Any, **kwargs: Any) -> str:
        raise TransportError(
            "create leg refused over REST: no create endpoint is verified on this "
            "instance; take the claim with `board.py upsert --live`, create through "
            "MCP operately.create_task, then re-run upsert to verify uniqueness"
        )


# ---------------------------------------------------------------------------
# The atomic lookup-or-create
# ---------------------------------------------------------------------------


def holders_of(doc: dict[str, Any], project_id: str, key: str) -> list[dict[str, Any]]:
    """Every row in one project that answers to this key. O(rows): one pass, one
    bounded identity computation per row."""
    return [
        task
        for task in doc["tasks"]
        if key in row_identities(task)
        and ((task.get("project_id") or "") or "").lower() == (project_id or "").lower()
    ]


def _resolve_project_id(doc: dict[str, Any], project: str) -> str:
    """The project's id, accepting a name, an id or an id prefix. O(projects)."""
    if not project:
        return ""
    lowered = project.strip().lower()
    for entry in doc["projects"]:
        if lowered in ((entry.get("name") or "").lower(), (entry.get("id") or "").lower()):
            return entry.get("id") or entry.get("name") or ""
    for entry in doc["projects"]:
        if (entry.get("id") or "").lower().startswith(lowered + "-"):
            return entry["id"]
    return project


def _outcome(action: str, **fields: Any) -> dict[str, Any]:
    out = {"action": action, "findings": []}
    out.update(fields)
    return out


def lookup_or_create(
    transport: Any,
    store: ClaimStore,
    *,
    project: str = "",
    name: str,
    description: str = "",
    explicit_key: str = "",
    milestone_id: str | None = None,
    apply: bool = False,
    successes: Iterable[str] = (),
    now: float | None = None,
    lease_s: int = LEASE_S,
    holder: str = "",
    doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a key against the whole board and create it only if absent.

    The claim is taken before the board is read, and released after the created
    row has been read back, so the check and the create are one critical section
    for every caller using this store. O(rows) per read, two or three reads.

    Returns an ``Outcome`` dict: ``action`` is one of

    ``existing``     the key was on the board; ``task_id`` is that row and
                     nothing was created.
    ``created``      exactly one row holds the key afterwards, and it is ours.
    ``would-create`` a dry run: absent, and ``--apply`` was not passed.
    ``duplicate``    the board already holds more than one row for this key.
    ``refused``      a finding names why; ``rule`` holds its name.

    ``successes`` names the closed row a new row succeeds. Without it, a closed
    holder is refused: a closed row is immutable, so follow-up work becomes a
    linked successor under its own scope rather than a reopened row.

    Raises ``TransportError`` when no project is named: identity is
    ``(project, key)``, and an unscoped lookup would match another project's row
    or miss this one's.
    """
    if not (project or "").strip():
        raise TransportError(
            "a creation key is scoped per project (identity is (project, key)); "
            "pass --project NAME_OR_ID"
        )
    now = time.time() if now is None else now
    key = creation_key(name, explicit_key)
    base_key = key
    board_doc = doc if doc is not None else transport.board()
    project_id = _resolve_project_id(board_doc, project)
    scope = scope_of(project_id or project, key)
    claim, why = store.acquire(scope, holder=holder or f"{project or 'space'}:{key}", now=now, lease_s=lease_s)
    if claim is None:
        return _outcome(
            "refused",
            rule=RULE_KEY_CLAIM_HELD,
            key=key,
            scope=scope,
            task_id="",
            read_back_count=0,
            detail=f"the claim on {scope} is held by another creator and is neither "
            "expired nor abandoned; a second creator would race the first, so it is refused",
        )
    try:
        # Read the board *inside* the claim: the twin that check-then-create
        # misses is created between the two, not during them.
        board_doc = transport.board()
        found = holders_of(board_doc, project_id, key)
        if len(found) > 1:
            return _outcome(
                "duplicate",
                rule=RULE_DUPLICATE_KEY,
                key=key,
                scope=scope,
                read_back_count=len(found),
                task_ids=[t["id"] for t in found],
                detail=f"{len(found)} rows on the board already hold key {key!r} "
                f"({', '.join(t['id'] for t in found)}); this path creates nothing and "
                "reports the twins for a lead to merge",
            )
        if found:
            row = found[0]
            if status_bucket(row) == "done":
                if row["id"] not in set(successes):
                    return _outcome(
                        "refused",
                        rule=RULE_CLOSED_ROW_IMMUTABLE,
                        key=key,
                        scope=scope,
                        task_id=row["id"],
                        read_back_count=1,
                        detail=f"row {row['id']} holds key {key!r} and is closed; a closed "
                        "row is immutable for landing, so follow-up work is a linked "
                        f"successor: pass --successes {row['id']}",
                    )
                # The successor is a new row under its own key and scope: the
                # closed row is never rewritten and never reopened.
                generation = len(found) + 1
                key = f"{base_key}-S{generation}"
                scope = successor_scope(scope_of(project_id or project, base_key), generation)
                store.release(claim)
                claim, why = store.acquire(
                    scope, holder=f"{project or 'space'}:{key}", now=now, lease_s=lease_s
                )
                if claim is None:
                    return _outcome(
                        "refused",
                        rule=RULE_KEY_CLAIM_HELD,
                        key=key,
                        scope=scope,
                        task_id="",
                        read_back_count=0,
                        detail=f"the successor scope {scope} is claimed by another creator",
                    )
                found = []
        if found:
            return _outcome(
                "existing",
                key=key,
                scope=scope,
                task_id=found[0]["id"],
                read_back_count=1,
                detail=f"key {key!r} is already on the board as {found[0]['id']}; "
                "nothing was created",
            )
        if not apply:
            return _outcome(
                "would-create",
                key=key,
                scope=scope,
                task_id="",
                read_back_count=0,
                detail=f"key {key!r} is absent from the whole board; pass --apply to create it",
            )
        body = description.rstrip() + f"\nKey: `{key}`."
        try:
            task_id = transport.create_task(project_id, name, body, milestone_id)
        except CreateOutcomeUnknown as exc:
            # The row may exist. The claim is released *without* recording an id,
            # so the retry re-reads the board inside a fresh claim and decides.
            return _outcome(
                "refused",
                rule=RULE_CREATE_UNKNOWN,
                key=key,
                scope=scope,
                task_id="",
                read_back_count=-1,
                detail=f"the create outcome is unknown ({exc}); the claim was released "
                "unrecorded, so re-run upsert: it reads the board inside the claim and "
                "returns the row if the write landed",
            )
        after = holders_of(transport.board(), project_id, key)
        if not after:
            return _outcome(
                "refused",
                rule="write-did-not-land",
                key=key,
                scope=scope,
                task_id=task_id,
                read_back_count=0,
                detail=f"create returned {task_id} but no row holding key {key!r} reads "
                "back; the create is not evidence",
            )
        if len(after) > 1:
            return _outcome(
                "duplicate",
                rule=RULE_DUPLICATE_KEY,
                key=key,
                scope=scope,
                task_id=task_id,
                read_back_count=len(after),
                task_ids=[t["id"] for t in after],
                detail=f"after creating {task_id}, {len(after)} rows hold key {key!r}: a "
                "creator outside this claim path produced a twin",
            )
        store.record(claim, after[0]["id"])
        return _outcome(
            "created",
            key=key,
            scope=scope,
            task_id=after[0]["id"],
            read_back_count=1,
            detail=f"created {after[0]['id']} and read back exactly one row holding key {key!r}",
        )
    finally:
        store.release(claim)


def render_outcome(outcome: dict[str, Any]) -> str:
    """One line per outcome field, the audit trail of a creation path."""
    head = f"upsert action={outcome['action']} key={outcome.get('key') or '-'}"
    if outcome.get("task_id"):
        head += f" task={outcome['task_id']}"
    if outcome.get("read_back_count") is not None:
        head += f" rows-holding-key={outcome['read_back_count']}"
    lines = [head]
    if outcome.get("rule"):
        lines.append(f"  rule: {outcome['rule']}")
    if outcome.get("task_ids"):
        lines.append(f"  twins: {', '.join(outcome['task_ids'])}")
    if outcome.get("detail"):
        lines.append(f"  {outcome['detail']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# In progress: a live lane lease or a cycle-stamped artifact change
# ---------------------------------------------------------------------------


def cycle_of(text: str) -> str:
    """The ``Cycle:`` date a row stamps its work with, or ``""``."""
    match = CYCLE_FIELD.search(flatten_description(text))
    if not match:
        return ""
    stamp = re.search(r"\b\d{4}-\d{2}-\d{2}\b", match.group(1))
    return stamp.group(0) if stamp else ""


def artifact_support(
    task: dict[str, Any], cycle: str, repo_root: str = "."
) -> dict[str, Any] | None:
    """A declared source path whose mtime date is not older than the stamped cycle.

    O(paths): one ``stat`` per declared path, each a single syscall.
    """
    if not cycle:
        return None
    try:
        boundary = date.fromisoformat(cycle)
    except ValueError:
        return None
    for raw in task.get("paths") or []:
        candidate = raw if os.path.isabs(raw) else os.path.join(repo_root, raw)
        try:
            stamp = datetime.fromtimestamp(os.stat(candidate).st_mtime, timezone.utc).date()
        except OSError:
            continue
        if stamp >= boundary:
            return {"path": raw, "cycle": cycle, "mtime": stamp.isoformat()}
    return None


def in_progress_support(
    task: dict[str, Any],
    store: ClaimStore | None = None,
    repo_root: str = ".",
    now: float | None = None,
) -> dict[str, Any]:
    """Whether an ``In progress`` row is evidence-bearing.

    Supported means a live lane lease naming this row's scope, or a cycle-stamped
    artifact change. O(paths) stats and one claim read.
    """
    scope = scope_of(task.get("project_id") or task.get("project_name") or "", row_key(task))
    lease = store.lease(scope, now=now) if store else {"present": False, "live": False}
    live_lease = bool(
        lease.get("live") and lease.get("task_id") in ("", task["id"])
    )
    cycle = cycle_of(task.get("text") or "")
    artifact = artifact_support(task, cycle, repo_root)
    supported = bool(live_lease or artifact)
    if live_lease:
        why = f"live lane lease on {scope} held by {lease.get('holder') or 'unknown'}"
    elif artifact:
        why = f"cycle {artifact['cycle']}: {artifact['path']} changed {artifact['mtime']}"
    elif cycle:
        why = f"cycle {cycle} stamped but no declared path changed on or after it"
    elif lease.get("present"):
        why = f"a claim exists on {scope} but it is not live for this row"
    else:
        why = "no lane lease and no cycle stamp"
    return {
        "task_id": task["id"],
        "name": task.get("name") or "",
        "scope": scope,
        "key": row_key(task),
        "status": task.get("status_value") or "",
        "supported": supported,
        "lease": lease if lease.get("present") else None,
        "cycle": cycle,
        "artifact": artifact,
        "detail": why,
    }


def lease_audit(
    doc: dict[str, Any],
    store: ClaimStore | None = None,
    repo_root: str = ".",
    now: float | None = None,
) -> list[dict[str, Any]]:
    """The predicate over every ``In progress`` row. O(rows) rows, each O(paths)."""
    rows = []
    for task in doc["tasks"]:
        if status_bucket(task) != "actionable" or (task.get("status_value") or "") != "in_progress":
            continue
        rows.append(in_progress_support(task, store, repo_root, now))
    return rows


def render_lease_audit(rows: list[dict[str, Any]], advisory: bool = True, detail: bool = True) -> str:
    """The predicate's table, plus its tally and its own enforcement state."""
    lines = []
    if detail:
        for row in rows:
            verdict = "supported" if row["supported"] else "UNSUPPORTED"
            lines.append(f"  {verdict}  {row['task_id']}  {row['detail']}")
    unsupported = [r for r in rows if not r["supported"]]
    lines.append(
        f"in-progress: rows={len(rows)} supported={len(rows) - len(unsupported)} "
        f"unsupported={len(unsupported)} "
        f"({'advisory; --strict reddens' if advisory else 'enforced'})"
    )
    return "\n".join(lines)


def cleanup(path: str) -> None:
    """Remove a scratch tree a control made. Best effort: a leftover copy is a
    cost, not a failure."""
    shutil.rmtree(path, ignore_errors=True)


__all__ = [
    "CREATION_RULES",
    "RULE_CLOSED_ROW_IMMUTABLE",
    "RULE_CREATE_UNKNOWN",
    "RULE_DUPLICATE_KEY",
    "RULE_KEY_CLAIM_HELD",
    "ClaimStore",
    "CreateOutcomeUnknown",
    "FileBoardTransport",
    "RestTransport",
    "artifact_support",
    "cleanup",
    "creation_key",
    "cycle_of",
    "declared_key",
    "holders_of",
    "in_progress_support",
    "lease_audit",
    "lookup_or_create",
    "normalize_key",
    "render_lease_audit",
    "render_outcome",
    "row_identities",
    "row_key",
    "scope_of",
    "scratch_dir",
]