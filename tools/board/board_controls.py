#!/usr/bin/env python3
"""board_controls: the seeded-defect suite behind ``--controls``.

Baseline green, every seeded defect red by name with the
subject it names, restored green, the check-in path confirmed by a
read-back, and the creation path confirmed by controls that each
create on a throwaway board. Nothing here touches the live board:
every creation runs against a JSON copy under a scratch directory.
Part of ``tools/board/board.py``.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from board_core import (  # noqa: F401
    TransportError,
    rules_of,
)
from board_check import (  # noqa: F401
    SEEDED_DEFECTS,
    _fixture_check,
    exit_code_for,
    post_offline,
)
from board_fixtures import (  # noqa: F401
    FIXTURE_BASELINE,
    baseline_document,
    describe,
)
from board_rows import (
    LEASE_S,
    ClaimStore,
    CreateOutcomeUnknown,
    FileBoardTransport,
    cleanup,
    creation_key,
    in_progress_support,
    lookup_or_create,
    row_key,
    scratch_dir,
    scope_of,
)

# ---------------------------------------------------------------------------
# Control suite
# ---------------------------------------------------------------------------


def run_controls(stream=None) -> int:
    """Baseline green, every seeded defect red by name, restored green."""
    stream = stream or sys.stdout
    failures = 0
    baseline = baseline_document()
    findings = _fixture_check(baseline)
    if findings:
        failures += 1
        print(
            f"CONTROL FAIL baseline: expected green, got {sorted(rules_of(findings))}", file=stream
        )
    else:
        print("CONTROL pass baseline: 0 findings", file=stream)

    for rule, mutate, subject in SEEDED_DEFECTS:
        doc = copy.deepcopy(baseline)
        mutate(doc)
        findings = _fixture_check(doc)
        rules = rules_of(findings)
        code = exit_code_for(findings)
        if rule not in rules:
            failures += 1
            print(
                f"CONTROL FAIL {rule}: seeded defect did not redden; got "
                f"{sorted(rules) or 'green'}",
                file=stream,
            )
            continue
        if code == 0:
            failures += 1
            print(f"CONTROL FAIL {rule}: findings reported but exit status was 0", file=stream)
            continue
        offending = [f for f in findings if f["rule"] == rule and subject not in f["detail"]]
        if offending:
            failures += 1
            print(
                f"CONTROL FAIL {rule}: finding does not name {subject!r}: "
                f"{offending[0]['detail']}",
                file=stream,
            )
            continue
        print(f"CONTROL pass {rule}: exit {code} on {subject}", file=stream)

    restored = baseline_document()
    if _fixture_check(restored):
        failures += 1
        print(f"CONTROL FAIL restore: expected green, got {sorted(rules_of(_fixture_check(restored)))}", file=stream)
    else:
        print("CONTROL pass restore: 0 findings", file=stream)

    # A reconciliation record must reach the board and be confirmed by a read-back.
    landed = post_offline(baseline, "apeiron", "green")
    if not landed:
        failures += 1
        print("CONTROL FAIL post-read-back: the record did not land on read-back", file=stream)
    else:
        print("CONTROL pass post-read-back: record confirmed by read-back", file=stream)
    refused = post_offline(baseline, "apeiron", "green", transport_applies=False)
    if refused:
        failures += 1
        print("CONTROL FAIL post-refusal: a write that did not land was accepted", file=stream)
    else:
        print("CONTROL pass post-refusal: unlanded write reported as not landed", file=stream)

    failures += run_creation_controls(stream)

    total = 1 + len(SEEDED_DEFECTS) + 1 + 2 + len(CREATION_CONTROLS)
    if failures:
        print(f"CONTROL FAIL suite: {failures} of {total} controls failed", file=stream)
        return 2
    print(f"CONTROL pass suite: {total}/{total} controls", file=stream)
    return 0


# ---------------------------------------------------------------------------
# Creation controls: an offline board, a private claim store, no network
# ---------------------------------------------------------------------------

APEIRON_ID = "apeiron-ExNAtv3hObMmq01Ab20DfA"
T0 = 1_767_225_600.0  # a fixed instant, so lease arithmetic is deterministic


def _board(root: str, raw: dict | None = None) -> FileBoardTransport:
    """A JSON board under the scratch root, seeded from the offline baseline."""
    path = os.path.join(root, "board.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(raw if raw is not None else copy.deepcopy(FIXTURE_BASELINE), handle)
    return FileBoardTransport(path, id_factory=lambda: "cafe0001")


def _upsert(transport, store, key: str, name: str, **kwargs):
    """One lookup-or-create against a throwaway board, project apeiron."""
    payload = {
        "project": "apeiron",
        "name": name,
        "description": f"Board: `{key}`.",
        "explicit_key": key,
        "apply": True,
        "now": T0,
        "lease_s": LEASE_S,
    }
    payload.update(kwargs)
    return lookup_or_create(transport, store, **payload)


def _holders(transport, key: str) -> list[str]:
    return [t["id"] for t in transport.board()["tasks"] if key in _identities(t)]


def _identities(task: dict) -> set[str]:
    from board_rows import row_identities

    return row_identities(task)


def control_existing_key_creates_nothing() -> tuple[bool, str]:
    """A create whose key is already on the board returns that row and creates none."""
    root = scratch_dir("board-existing-")
    try:
        transport, store = _board(root), ClaimStore(os.path.join(root, "state"))
        before = len(transport.board()["tasks"])
        out = _upsert(transport, store, "FIXTURE-E1-W1-T2", "G1/REG-1 acceptance green")
        after = len(transport.board()["tasks"])
        if out["action"] != "existing":
            return False, f"expected action=existing, got {out['action']} ({out.get('rule')})"
        if out["task_id"] != "g1-reg-1-acceptance-green-BBBB2222":
            return False, f"returned row {out['task_id']}, not the row already on the board"
        if after != before:
            return False, f"rows went {before} -> {after}: the existing path created a twin"
        return True, f"returned {out['task_id']}, rows {before} -> {after}"
    finally:
        cleanup(root)


def control_new_key_creates_exactly_one() -> tuple[bool, str]:
    """A new key creates exactly one row, read back, and a re-run returns it."""
    root = scratch_dir("board-new-")
    try:
        transport, store = _board(root), ClaimStore(os.path.join(root, "state"))
        before = len(transport.board()["tasks"])
        out = _upsert(transport, store, "APEIRON-BOARD-W1-T9", "Board driver wave 1 task 9")
        created = out["task_id"]
        holders = _holders(transport, "APEIRON-BOARD-W1-T9")
        if out["action"] != "created":
            return False, f"expected action=created, got {out['action']} ({out.get('rule')})"
        if len(transport.board()["tasks"]) != before + 1:
            return False, f"rows went {before} -> {len(transport.board()['tasks'])}, not +1"
        if holders != [created]:
            return False, f"expected exactly one row holding the key, got {holders}"
        again = _upsert(transport, store, "APEIRON-BOARD-W1-T9", "Board driver wave 1 task 9")
        if again["action"] != "existing" or again["task_id"] != created:
            return False, f"re-run gave {again['action']}/{again['task_id']}, not existing/{created}"
        if _holders(transport, "APEIRON-BOARD-W1-T9") != [created]:
            return False, "the re-run produced a twin"
        return True, f"created {created}, one holder, re-run returned it"
    finally:
        cleanup(root)


class _KilledTransport:
    """A create that landed, after which the process died before recording it.

    The write is dispatched and then the connection is lost, which the transport
    reports as ``CreateOutcomeUnknown`` because it cannot tell whether the row
    exists. That is the state a lane killed mid-create leaves behind.
    """

    def __init__(self, inner: FileBoardTransport) -> None:
        self.inner = inner

    def board(self):
        return self.inner.board()

    def create_task(self, *args, **kwargs) -> str:
        task_id = self.inner.create_task(*args, **kwargs)
        raise CreateOutcomeUnknown("connection reset after the write was dispatched")


def control_killed_create_leaves_no_twin() -> tuple[bool, str]:
    """A killed create, then a retry: the retry returns the landed row, no twin.

    Two halves. With the claim released, the retry takes a fresh claim and finds
    the row the dead write left behind. With a claim still on disk and no id in it
    -- the hard death, where nothing ran the release -- a retry inside the lease is
    refused, and a retry after the lease takes the claim over and still finds the
    one row.
    """
    root = scratch_dir("board-killed-")
    try:
        transport, store = _board(root), ClaimStore(os.path.join(root, "state"))
        key = "APEIRON-BOARD-W2-T1"
        before = len(transport.board()["tasks"])
        killed = _upsert(_KilledTransport(transport), store, key, "Board driver wave 2 task 1")
        if killed["action"] != "refused" or killed.get("rule") != "create-outcome-unknown":
            return False, (
                f"a killed create reported {killed['action']}/{killed.get('rule')}, "
                "not refused/create-outcome-unknown"
            )
        retry = _upsert(transport, store, key, "Board driver wave 2 task 1", now=T0 + 1)
        if retry["action"] != "existing":
            return False, f"retry after a released claim gave {retry['action']}, not existing"
        if len(transport.board()["tasks"]) != before + 1:
            return False, "the retry after a killed create produced a twin"
        # Hard death: the claim is on disk with no row id recorded in it.
        claim, why = store.acquire(scope_of(APEIRON_ID, key), holder="dead-lane", now=T0 + 2)
        if claim is None or why != "held":
            return False, f"the abandoned claim was not taken ({why})"
        store._write(store.path_for(claim["scope"]), dict(claim, task_id=""))
        blocked = _upsert(transport, store, key, "Board driver wave 2 task 1", now=T0 + 3)
        if blocked.get("rule") != "creation-key-claim-held":
            return False, f"a retry inside the lease gave {blocked['action']}, not claim-held"
        after_lease = _upsert(
            transport, store, key, "Board driver wave 2 task 1", now=T0 + 3 + LEASE_S + 1
        )
        if after_lease["action"] != "existing":
            return False, f"a retry after the lease gave {after_lease['action']}, not existing"
        holders = _holders(transport, key)
        if len(holders) != 1:
            return False, f"expected one row holding {key}, got {holders}"
        return True, (
            f"killed create left {holders[0]}; retry returned it; "
            "inside-lease retry refused; after-lease retry took over and returned it"
        )
    finally:
        cleanup(root)


def control_live_claim_refuses_a_second_creator() -> tuple[bool, str]:
    """A live claim on the key refuses the second creator: the race has no window."""
    root = scratch_dir("board-claim-")
    try:
        transport, store = _board(root), ClaimStore(os.path.join(root, "state"))
        key = "APEIRON-BOARD-W3-T1"
        claim, why = store.acquire(scope_of(APEIRON_ID, key), holder="first-creator", now=T0)
        if claim is None:
            return False, f"the first creator could not take the claim ({why})"
        before = len(transport.board()["tasks"])
        out = _upsert(transport, store, key, "Board driver wave 3 task 1", now=T0 + 1)
        after = len(transport.board()["tasks"])
        if out.get("rule") != "creation-key-claim-held":
            return False, f"a second creator was not refused: {out['action']}/{out.get('rule')}"
        if after != before:
            return False, f"the refused creator still wrote a row ({before} -> {after})"
        return True, f"refused with creation-key-claim-held, rows {before} -> {after}"
    finally:
        cleanup(root)


def control_duplicate_key_is_reported() -> tuple[bool, str]:
    """A board already holding two rows for one key is reported, not resolved."""
    root = scratch_dir("board-dupe-")
    try:
        raw = copy.deepcopy(FIXTURE_BASELINE)
        twin = copy.deepcopy(raw["tasks"][1])
        twin["id"] = "g1-acceptance-twin-DDDD6666"
        raw["tasks"].append(twin)
        transport, store = _board(root, raw), ClaimStore(os.path.join(root, "state"))
        before = len(transport.board()["tasks"])
        out = _upsert(transport, store, "FIXTURE-E1-W1-T2", "G1/REG-1 acceptance green")
        after = len(transport.board()["tasks"])
        if out.get("rule") != "duplicate-creation-key":
            return False, f"a twin pair gave {out['action']}/{out.get('rule')}"
        if out.get("read_back_count") != 2:
            return False, f"expected 2 rows holding the key, read {out.get('read_back_count')}"
        if after != before:
            return False, f"the duplicate path created another row ({before} -> {after})"
        return True, "reported 2 rows holding the key and created nothing"
    finally:
        cleanup(root)


def control_closed_row_needs_a_linked_successor() -> tuple[bool, str]:
    """A closed holder is refused, then succeeded by a new row, never reopened."""
    root = scratch_dir("board-closed-")
    try:
        transport, store = _board(root), ClaimStore(os.path.join(root, "state"))
        key = "FIXTURE-E1-W3-T1"
        closed = "receipt-archive-complete-DDDD4444"
        before = len(transport.board()["tasks"])
        refused = _upsert(transport, store, key, "Archive the acceptance receipt")
        if refused.get("rule") != "closed-row-needs-linked-successor":
            return False, f"a closed holder gave {refused['action']}/{refused.get('rule')}"
        if refused.get("task_id") != closed:
            return False, f"the refusal did not name the closed row {closed}"
        successor = _upsert(
            transport, store, key, "Archive the acceptance receipt, take 2",
            successes=[closed],
        )
        doc = transport.board()
        after = [t for t in doc["tasks"] if t["id"] == closed][0]
        if successor["action"] != "created":
            return False, f"the linked successor gave {successor['action']}/{successor.get('rule')}"
        if len(doc["tasks"]) != before + 1:
            return False, f"rows went {before} -> {len(doc['tasks'])}, not +1"
        if after["status_value"] != "done":
            return False, "the closed row was reopened rather than succeeded"
        if not successor["key"].startswith(key + "-S"):
            return False, f"the successor reused the closed row's key {successor['key']!r}"
        return True, f"refused, then created {successor['task_id']} under key {successor['key']}"
    finally:
        cleanup(root)


def _stamped_artifact(root: str, days_ago: int) -> tuple[str, str]:
    """A declared source path whose mtime is ``days_ago`` days back, and its date."""
    path = os.path.join(root, "src", "lib.rs")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("// control artifact\n")
    stamp = (datetime.now(timezone.utc) - timedelta(days=days_ago)).replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    os.utime(path, (stamp.timestamp(), stamp.timestamp()))
    return path, stamp.date().isoformat()


def _in_progress_row(task_id: str, body: str) -> dict:
    """One ``In progress`` row, isolated from the rest of the baseline."""
    doc = baseline_document()
    task = [t for t in doc["tasks"] if t["id"] == task_id][0]
    task["text"] = body
    task["paths"] = ["src/lib.rs"]
    task["status_value"] = "in_progress"
    task["status_label"] = "In progress"
    return task


def control_in_progress_unsupported_is_reported() -> tuple[bool, str]:
    """An In progress row with no lease and no artifact change is unsupported."""
    root = scratch_dir("board-lease-")
    try:
        store = ClaimStore(os.path.join(root, "state"))
        now = T0 + 10_000
        bare = _in_progress_row(
            "g1-reg-1-acceptance-green-BBBB2222",
            describe(
                "Board: `FIXTURE-E1-W1-T2`.",
                "Gate: `cargo test --workspace --all-features --locked`.",
                "Source: `src/lib.rs`.",
            ),
        )
        if in_progress_support(bare, store, repo_root=root, now=now)["supported"]:
            return False, "a row with no lease and no cycle stamp was reported supported"
        path, _ = _stamped_artifact(root, days_ago=10)
        today = datetime.now(timezone.utc).date().isoformat()
        stale = _in_progress_row(
            "g1-reg-1-acceptance-green-BBBB2222",
            describe(
                "Board: `FIXTURE-E1-W1-T2`.",
                f"Cycle: {today}.",
                "Gate: `cargo test --workspace --all-features --locked`.",
                "Source: `src/lib.rs`.",
            ),
        )
        outcome = in_progress_support(stale, store, repo_root=root, now=now)
        if outcome["supported"]:
            return False, f"cycle {today} was accepted for an artifact last changed at {path}"
        if "changed on or after" not in outcome["detail"]:
            return False, f"the unsupported row did not name the missing change: {outcome['detail']}"
        return True, "no lease + no stamp, and a stamp newer than its artifact, both unsupported"
    finally:
        cleanup(root)


def _dead_pid() -> int:
    """A pid that has certainly exited: a reaped child."""
    child = subprocess.Popen(["/bin/true"])
    child.wait()
    return child.pid


def control_in_progress_supported_legs() -> tuple[bool, str]:
    """The positive legs: a live lane lease, and a cycle-stamped artifact change."""
    root = scratch_dir("board-lease-ok-")
    try:
        store = ClaimStore(os.path.join(root, "state"))
        now = T0 + 10_000
        task = _in_progress_row(
            "g1-reg-1-acceptance-green-BBBB2222",
            describe(
                "Board: `FIXTURE-E1-W1-T2`.",
                "Gate: `cargo test --workspace --all-features --locked`.",
                "Source: `src/lib.rs`.",
            ),
        )
        scope = scope_of(task["project_id"], row_key(task))
        claim, why = store.acquire(scope, holder="lane-apeiron-board", now=now)
        if claim is None:
            return False, f"the lane could not take its lease on {scope} ({why})"
        if not in_progress_support(task, store, repo_root=root, now=now)["supported"]:
            return False, "a live lane lease was not accepted as evidence"
        # A claim whose holder is gone is not a live lane.
        store._write(store.path_for(scope), dict(claim, pid=_dead_pid(), task_id=""))
        if in_progress_support(task, store, repo_root=root, now=now)["supported"]:
            return False, "a claim held by a dead pid was accepted as a live lane"
        store.release(claim)
        today = datetime.now(timezone.utc).date().isoformat()
        fresh = _in_progress_row(
            "g1-reg-1-acceptance-green-BBBB2222",
            describe(
                "Board: `FIXTURE-E1-W1-T2`.",
                f"Cycle: {today}.",
                "Gate: `cargo test --workspace --all-features --locked`.",
                "Source: `src/lib.rs`.",
            ),
        )
        _stamped_artifact(root, days_ago=0)
        outcome = in_progress_support(fresh, store, repo_root=root, now=now)
        if not outcome["supported"]:
            return False, f"a cycle {today} artifact change was not accepted: {outcome['detail']}"
        return True, "live lease accepted, dead-holder claim refused, cycle artifact accepted"
    finally:
        cleanup(root)


def control_key_rule_is_stable() -> tuple[bool, str]:
    """The key rule: one token per spelling, an explicit key beats the name."""
    cases = [
        (creation_key("anything", "APEIRON-BOARD-W1-T2"), "APEIRON-BOARD-W1-T2"),
        (creation_key("Board driver — wave 1 task 9"), "BOARD-DRIVER-WAVE-1-TASK-9"),
        (creation_key("G1/REG-1 acceptance green"), "G1-REG-1"),
        (creation_key("g1 reg 1 acceptance green"), "G1-REG-1-ACCEPTANCE-GREEN"),
    ]
    for got, want in cases:
        if got != want:
            return False, f"creation_key gave {got!r}, expected {want!r}"
    if row_key({"text": "Key: `APEIRON-BOARD-W1-T2`.", "name": "Renamed", "plan_ids": []}) != (
        "APEIRON-BOARD-W1-T2"
    ):
        return False, "a declared Key: field did not win over the display name"
    root = scratch_dir("board-key-")
    try:
        # Identity is (project, key): an unscoped lookup is refused rather than
        # matching another project's row.
        try:
            lookup_or_create(
                _board(root),
                ClaimStore(os.path.join(root, "state")),
                project="",
                name="No project",
                explicit_key="APEIRON-BOARD-W1-T2",
                apply=True,
                now=T0,
            )
        except TransportError:
            pass
        else:
            return False, "an upsert with no project was accepted; identity is (project, key)"
    finally:
        cleanup(root)
    return True, f"{len(cases)} spellings normalize to one key each; Key: beats the title"


CREATION_CONTROLS = (
    ("creation-key-rule", control_key_rule_is_stable),
    ("lookup-or-create-existing", control_existing_key_creates_nothing),
    ("lookup-or-create-new", control_new_key_creates_exactly_one),
    ("lookup-or-create-atomic-retry", control_killed_create_leaves_no_twin),
    ("lookup-or-create-claim-held", control_live_claim_refuses_a_second_creator),
    ("lookup-or-create-duplicate-key", control_duplicate_key_is_reported),
    ("closed-row-linked-successor", control_closed_row_needs_a_linked_successor),
    ("in-progress-unsupported", control_in_progress_unsupported_is_reported),
    ("in-progress-supported", control_in_progress_supported_legs),
)


def run_creation_controls(stream=None) -> int:
    """Every creation control must pass; returns the failure count.

    O(controls): each makes one scratch board under ``/tmp/opencode`` and removes
    it, so the suite reads no repository file and writes no live board.
    """
    stream = stream or sys.stdout
    failures = 0
    for name, control in CREATION_CONTROLS:
        try:
            ok, detail = control()
        except Exception as exc:  # noqa: BLE001 - a broken control is a failure
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        if ok:
            print(f"CONTROL pass {name}: {detail}", file=stream)
        else:
            failures += 1
            print(f"CONTROL FAIL {name}: {detail}", file=stream)
    return failures