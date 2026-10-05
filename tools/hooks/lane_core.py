#!/usr/bin/env python3
"""lane_core: the shared vocabulary of the board-declaration hook.

`tools/hooks/lane.py` is the entry point the git hooks call
(``python3 tools/hooks/lane.py staged|message|...``). It dispatches; the
implementation lives in these sibling modules, imported by name, so the command
surface, the diagnostics and the exit codes are unchanged:

  * ``lane_core``      rule names, the staged-change readers, board resolution,
                       row matching and the ``Commit`` object.
  * ``lane_fetch``     exhaustive board enumeration and the live client.
  * ``lane_modes``     the modes the CLI dispatches to.
  * ``lane_controls``  ``self-test``: the controls.

The ``board:`` rule is not restated here: ``tools/board/board.py`` is the
authority for the field and is imported, not re-implemented (module ``board``).
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
from typing import Any
from typing import Iterable

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

# The commit hook's wall-clock budget, shared by .githooks/commit-msg (which
# wraps lane.py in `timeout "$APEIRON_HOOKS_COMMITMSG_BUDGET"`) and the
# enumeration quorum in lane_fetch. A busy board is re-enumerated until two
# *consecutive* reads agree; when the internal deadline passes the fetch refuses
# by name rather than resolving against a partial read. ``QUORUM_HEADROOM`` is
# reserved inside that budget so the named refusal is printed before the hook's
# own ``timeout`` sends SIGTERM -- a SIGTERM would report exit 124 and lose the
# rule, turning a named refusal into an unexplained budget failure.
COMMITMSG_BUDGET_ENV = "APEIRON_HOOKS_COMMITMSG_BUDGET"
COMMITMSG_BUDGET_DEFAULT = 90.0
QUORUM_HEADROOM = 5.0

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


class FetchError(RuntimeError):
    """A named enumeration refusal; ``rule`` is what the hook prints."""

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail


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


def commitmsg_budget() -> float:
    """The enumeration quorum's wall-clock budget, in seconds.

    Read from ``APEIRON_HOOKS_COMMITMSG_BUDGET`` (default 90 s). A value that is
    absent, unparseable or non-positive yields the default: a zero budget would
    refuse every board -- including the quiesced one this retry exists to
    accept -- which is the failure mode, not a stronger gate.
    """
    raw = os.environ.get(COMMITMSG_BUDGET_ENV, "").strip()
    if not raw:
        return COMMITMSG_BUDGET_DEFAULT
    try:
        value = float(raw)
    except ValueError:
        return COMMITMSG_BUDGET_DEFAULT
    return value if value > 0 else COMMITMSG_BUDGET_DEFAULT


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
