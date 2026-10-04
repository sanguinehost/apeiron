#!/usr/bin/env python3
"""board_fixtures: the offline baseline the control suite reads.

``--fixtures`` makes a run hermetic: the inline plan is the
source of truth, so no repository file is read behind the fixture's back, and
descriptions are stored the way the live instance stores them (a ProseMirror
document in a JSON string), so the parser is exercised on the real shape. Part
of ``tools/board/board.py``.
"""

from __future__ import annotations

import copy
import json
from typing import Any
from board_core import (
    DEFAULT_SPACE_ID,
    SCHEMA,
)
from board_model import (
    attribute_projects,
    normalize_document,
)

# ---------------------------------------------------------------------------
# Fixtures (offline; the control suite's only input)
# ---------------------------------------------------------------------------


def describe(*paragraphs: str) -> str:
    """Encode text the way the instance stores a description: a ProseMirror doc.

    The live board holds every description as such a JSON document in a JSON
    string, so the fixtures do too: the parser is exercised on the real shape.
    """
    content = [
        {"content": [{"text": p, "type": "text"}], "type": "paragraph"}
        if p
        else {"type": "paragraph"}
        for p in paragraphs
    ]
    return json.dumps({"content": content, "type": "doc"})


APEIRON = {"id": "apeiron-ExNAtv3hObMmq01Ab20DfA", "name": "apeiron"}
SONG = {"id": "song-Bl3Jgp65PHjxnpmdCpxBoH", "name": "song"}


def _task(
    task_id: str,
    name: str,
    status_value: str,
    status_label: str,
    body: str,
    project: dict[str, str] | None = None,
    milestone: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    task = {
        "id": task_id,
        "name": name,
        "status": {"value": status_value, "label": status_label},
        "description": body,
        "project": project,
        "milestone": {"id": milestone} if milestone else None,
    }
    task.update(extra)
    return task


FIXTURE_PLAN_MD = """# Fixture execution plan

## Phasing: epics, waves, tasks

### E1 — Board driver (Rust, `tools/board`)

| wave | task | files | gate | control | agents | heavy |
|---|---|---|---|---|---|---|
| W1 | `FIXTURE-E1-W1-T1` rooting gate green: the binned routes are red | `tools/rooting/check.py` | `python3 tools/rooting/check.py --check` | binned-route mutation red, restore green | 1 | 0 |
| W1 | `FIXTURE-E1-W1-T2` G1/REG-1 acceptance green | `examples/reg1/acceptance.rs` | `cargo test --workspace --all-features --locked` | strict acceptance exits 1 | 1 | 1 |
| W3 | `FIXTURE-E1-W3-T1` archive the acceptance receipt — **landed** | `docs/evidence/` | `python3 tools/board/board.py --check --controls` | seeded defects red by name | 1 | 0 |
"""

# song's ledger carries song's rows only: a shared file mixing two projects'
# rows makes a project-scoped run report the other project's rows as drift, and
# the fixture is the model the live run follows.
FIXTURE_LEDGER_TSV = (
    "id\ttask\thorizon\tstate\tpaths\tgate\treceipt\tresidual\n"
    "W0-GATE\tWire the gate adapter\tsprint\tclosed\tREADME.md\tits gate\t"
    "artifacts/W0-GATE/receipt.txt\tadapter only\n"
    "W1-I01\tSeed emit instruction inventory\tsprint\topen\tsrc/emit/x86.slang\t"
    "`tools/gate hold`\t\tseed compiler assembly\n"
    "W2-T1\tLand GATE-WAVE-3 and the freeze corrections\tsprint\topen\t"
    "song-ref/src/gate/main.slang\t`tools/gate hold`\t\tfreeze corrections\n"
    "W2-T2\tWire the kernel gate subcommand\tsprint\topen\t"
    "song-ref/src/gate/kernel.slang\t`tools/gate hold`\t\tkernel gate subcommand\n"
    "R1-T1\tFixture lane row\tsprint\topen\tsong-ref/src/gate/manifest.slang\t"
    "`tools/gate hold`\t\tresume queue, mirror probe\n"
)

FIXTURE_BASELINE: dict[str, Any] = {
    "schema": SCHEMA,
    "origin": "fixtures:baseline",
    "space": {"id": DEFAULT_SPACE_ID, "name": "sanguine"},
    "projects": [APEIRON, SONG],
    "milestones": [
        {"id": "m1-assembly-core", "title": "M1 — Assembly core complete", "project": SONG},
        {"id": "m2-kernel", "title": "M2 — Kernel and gate wiring", "project": SONG},
    ],
    "plan": {
        "sources": [
            {"path": "docs/plan.md", "project": "apeiron", "text": FIXTURE_PLAN_MD},
            {"path": "plan/ledger.tsv", "project": "song", "text": FIXTURE_LEDGER_TSV},
        ]
    },
    "tasks": [
        _task(
            "rooting-gate-green-AAAA1111",
            "Rooting gate green",
            "pending",
            "Not started",
            describe(
                "Board: `FIXTURE-E1-W1-T1`.",
                "Plan row `FIXTURE-E1-W1-T1`.",
                "`python3 tools/rooting/check.py --check` is red on the binned routes.",
                "Source: `docs/plan.md` §Canonical root and scope.",
            ),
            project=APEIRON,
        ),
        _task(
            "g1-reg-1-acceptance-green-BBBB2222",
            "G1/REG-1 acceptance green",
            "in_progress",
            "In progress",
            describe(
                "Board: `FIXTURE-E1-W1-T2`.",
                "Plan row `FIXTURE-E1-W1-T2`.",
                "Strict acceptance must exit 0 with `full_acceptance=true`.",
                "Source: `docs/plan.md`; `docs/evidence/perf-ledger.md`.",
                "Gate: `cargo test --workspace --all-features --locked`.",
            ),
            project=APEIRON,
        ),
        _task(
            "land-gate-wave-3-CCCC3333",
            "Land GATE-WAVE-3 and the freeze corrections",
            "in_progress",
            "In progress",
            describe(
                "Board: `W2-T1`.",
                "Plan row `W2-T1`.",
                "Freeze corrections at revision 9.",
                "Source: `README.md`.",
                "Gate: `cargo test --workspace --all-features --locked`.",
            ),
            project=SONG,
            milestone="m1-assembly-core",
        ),
        _task(
            "kernel-gate-wiring-EEEE5555",
            "Wire the kernel gate subcommand",
            "pending",
            "Not started",
            describe(
                "Board: `W2-T2`.",
                "Plan row `W2-T2`.",
                "The kernel gate subcommand is missing.",
                "Source: `plan/ledger.tsv`.",
                "Gate: `tools/gate hold`.",
            ),
            project=SONG,
            milestone="m2-kernel",
        ),
        _task(
            "read-gate-manifest-FFFF6666",
            "Fixture lane row",
            "pending",
            "Not started",
            describe(
                "Board: `R1-T1`.",
                "Plan row `R1-T1`.",
                "Read the gate binary manifest and report.",
                "Source: `tools/gate/Cargo.toml`.",
                "Gate: `tools/gate hold`.",
            ),
            project=SONG,
        ),
        _task(
            "toolchain-probe-9999AAAA",
            "Probe the pinned toolchain versions",
            "pending",
            "Not started",
            describe(
                "Board: none -- internal lane.",
                "Internal lane, internal-by-design: read the pinned versions and report, "
                "no board row expected.",
                "Source: `proof/README.md`.",
                "Gate: `proof/regen.sh --check`.",
            ),
            project=SONG,
        ),
        _task(
            "seed-emit-inventory-8888BBBB",
            "Seed emit instruction inventory",
            "in_progress",
            "In progress",
            describe(
                "Board: `W1-I01`.",
                "Plan row `W1-I01`.",
                "Reusable instruction and refusal fragments; the compiler import is refused.",
                "Source: `plan/ledger.tsv`; `song-ref/src/emit/x86.slang`.",
                "Gate: `tools/gate hold`.",
            ),
            project=SONG,
        ),
        _task(
            "receipt-archive-complete-DDDD4444",
            "Archive the acceptance receipt",
            "done",
            "Done",
            describe(
                "Board: `FIXTURE-E1-W3-T1`.",
                "Plan row `FIXTURE-E1-W3-T1`.",
                "Receipt written and bound to the recorded source digest "
                "sha256=e3b0c44298fc1c149afbf4c8996fb924.",
                "Source: `docs/evidence/locator-reserve-quadratic-and-reg1-attribution.md`.",
                "Gate: `python3 tools/board/board.py --check --controls`.",
                "Receipt: `docs/evidence/locator-reserve-quadratic-and-reg1-attribution.md`.",
                "Due: 2026-10-05, from the receipt above.",
            ),
            project=APEIRON,
        ),
    ],
}


def baseline_document() -> dict[str, Any]:
    doc = normalize_document(copy.deepcopy(FIXTURE_BASELINE))
    attribute_projects(doc)
    return doc
