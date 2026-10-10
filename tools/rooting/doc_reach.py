#!/usr/bin/env python3
"""Reachability gate for documentation removals.

Row: `APEIRON-ENFORCER-REACHABILITY-2` (`apeiron-enforcer-2-the-DoniUgsOm7EBIMgZWNTZWn`).

`AGENTS.md` §1a records this gate as **MISSING**: *"a documentation-removal
reachability gate for the documentation consolidations (the evidence-receipt half
now runs as `tools/rooting/index_check.py`)"*. `AGENTS.md` §10 (failure register,
row *Consolidation lost files*) names the obligation the gate discharges: a
removal is a migration or an authorised retirement, and it must **name its
replacement**, **inventory its affected consumers** and **gate reachability** —
"a generic reachability gate for removals; every removal requires its consumer
inventory and its task-specific reachability gate now".

`tools/rooting/retire_check.py` rules M1–M4 read a removal *record* and refuse a
record whose fields are missing, unresolvable or unbacked. That is a check on the
record. This gate is the check on the **tree**: for every removal the repository
registers in its own migration table `docs/moved.tsv` (one
``removed<TAB>successor`` row per line, the tree's own record of the
documentation consolidations), it resolves the successor, confirms the removed
path is gone, and computes the **consumer inventory** — the live documents that
still name the removed path — and refuses while that inventory is non-empty.

Three legs, one per acceptance requirement of the §10 row:

* (a) a named replacement — `D1` refuses a row with no successor, `D2` refuses a
  successor that does not resolve to a file under the root (or names the removed
  path itself);
* (b) an inventory of consumers — computed by the scan, not declared: `--check`
  prints it per removal, `--inventory PATH` prints it for one path and exits 1
  while it is non-empty;
* (c) a reachability check that fails closed — `D4` refuses while any live
  document in the documentation surface still names a removed path, printing the
  consumer, the line, the removed path and the successor that took over.

`D3` is the fourth rule: a registered removal whose removed path is still on disk
is not a removal, so the table is stale in the other direction.

Rules
=====
* ``S`` surface: the registry or the documentation root is absent;
* ``D1`` no successor: a row names no replacement;
* ``D2`` unreachable successor: the successor resolves to no file under the
  scanned root, or names the removed path;
* ``D3`` removed path present: the removed path still exists in the tree;
* ``D4`` live consumer: a document in the surface names a removed path on a line
  that does not also name the successor (see the migration-note exemption);
* ``D5`` registry row structure: a row that is not exactly one tab-separated
  pair, an empty or absolute or root-escaping path, or a duplicate removed path —
  a **data error** (exit 2), because neither the id set nor the consumer map can
  be established without them. Exit 1 is content, exit 2 is structure.

What is checked
===============
The **documentation surface** is every ``*.md`` file in the scanned tree except
those under an excluded directory: ``.git``, ``target``, ``__pycache__``,
``*.lake``, ``.tracks``, ``.agents``, ``.claude``, ``.codex``, ``.aws``,
``.licenses``, ``.held-control``, ``.held-scratch``. Those directories are
untracked (``git ls-files`` returns nothing for each) and each declares itself
non-source — ``.tracks/README.md`` says "Working notes only. Gitignored. Not
source, not evidence, not committed." — so excluding them keeps the verdict a
function of tree state rather than of one operator's local scratch.

A **reference** is a whole-token occurrence of a registered removed path in the
text of a surface document, reported with its line number. There is exactly one
migration-note exemption: the line also names the row's **successor**
(``docs/old/intake.md`` is now ``docs/intent.md``). Naming both paths is the
checkable way to record a preimage, so the rule accepts a migration record
without a lexical heuristic, and refuses everything else — including a reference
*from the successor document itself*, because on this tree the only two
references in the whole surface are the stale raw-log table in
``docs/evidence/perf-ledger.md``. Exemptions are printed by ``--list``.

Trust boundary
==============
The gate reads the registry and the documentation surface, and it writes nothing
under ``--check``. It is therefore evidence that **registered** removals name a
resolvable successor and are no longer referenced from a document — nothing
more. It cannot see a documentation removal that was never registered: there is
no row to refuse and no baseline of the previous tree to diff, the same exclusion
``retire_check.py`` states for M1–M4. On this tree that population is not empty:
a Markdown scan for repo-relative paths that resolve to nothing reports
``docs/extraction-requirements.md`` (named by ``docs/README.md`` and
``docs/archive/handoff-openai.md``) and ``docs/apeiron-intent-escalation.md``
(named by ``docs/plan.md``), neither of which carries a ``docs/moved.tsv`` row.
Closing that needs either a git-baseline discovery leg or a registry row per
unregistered removal; both are outside this gate and are recorded as the open
obligation. The surface is Markdown only: a tool, a receipt or a removal record
that names a removed path is a record of the removal, not a consumer, and is
outside the surface — so a live *program* reading a removed document is not seen
here. Resolution is existence, not correctness: a successor that resolves is not
shown to carry the removed content. Symlinks are followed. No network, no clock,
no Lean, no Rust.

Isolation
=========
``--self-test`` and ``--controls`` materialise trees under the system temporary
directory; the working tree is read-only and never mutated. ``--self-test``
never reads the repository. ``--controls`` copies the real ``docs/`` subtree,
runs baseline / mutation / restore there, and removes every copy on exit. No
fixture depends on a path that exists in the real tree. The system temporary
directory honours ``TMPDIR`` — set ``TMPDIR=/home/socol/oc-tmp`` to keep copies
off the ``/tmp`` tmpfs, which has run out of space mid-campaign.

Usage::

    python3 tools/rooting/doc_reach.py --list
    python3 tools/rooting/doc_reach.py --self-test
    python3 tools/rooting/doc_reach.py --controls
    python3 tools/rooting/doc_reach.py --check
    python3 tools/rooting/doc_reach.py --inventory docs/plan/iota-source-refinement.md

Every mode accepts ``--root PATH`` to scan another tree, which is what the
controls use, and ``--registry PATH`` to read another migration table.

Exit status: 0 when every registered removal names a resolvable successor, its
removed path is absent and no document in the surface names it; 1 when a rule
fires; 2 on a usage or registry-structure error.
"""

from __future__ import annotations

import argparse
import bisect
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]

EXIT_OK = 0
EXIT_REJECT = 1
EXIT_USAGE = 2

REGISTRY = "docs/moved.tsv"
DOCS_ROOT = "docs"
DOC_SUFFIX = ".md"

# Untracked working/tool directories: excluded so the verdict does not depend on
# one operator's local scratch. See the module docstring.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        "target",
        "__pycache__",
        ".lake",
        ".tracks",
        ".agents",
        ".claude",
        ".codex",
        ".aws",
        ".licenses",
        ".held-control",
        ".held-scratch",
    }
)

# A reference is a whole-token occurrence of the path: no word character may sit
# next to it. `docs/plan/x.md` must not match inside `x.md.bak`, and the sentence
# period in `see docs/plan/x.md.` must not suppress the reference.
WORD = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


def _is_whole_token(text: str, at: int, length: int) -> bool:
    """Whether ``text[at:at+length]`` is a whole-token occurrence."""
    before = text[at - 1] if at else ""
    if before in WORD:
        return False
    tail = at + length
    after = text[tail] if tail < len(text) else ""
    if after in WORD or after == "/":
        return False
    if after == ".":
        # `x.md.bak` and `x.md5` continue the token; `x.md.` ends a sentence.
        if tail + 1 < len(text) and text[tail + 1] in WORD:
            return False
    return True


class GateError(Exception):
    """The gate cannot reach its acceptance condition."""


class DataError(GateError):
    """The migration table is malformed."""


@dataclass(frozen=True)
class Removal:
    removed: str
    successor: str
    line: int


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    line: int
    message: str

    def render(self) -> str:
        where = self.path if self.line <= 0 else f"{self.path}:{self.line}"
        return f"RULE {self.rule} {where}: {self.message}"


@dataclass(frozen=True)
class Summary:
    rows: int
    documents: int
    gated: int
    non_markdown: int
    consumers: int
    notes: int

    def render(self) -> str:
        return (
            f"  rows={self.rows} documents={self.documents} "
            f"gated={self.gated} non_markdown={self.non_markdown} "
            f"consumers={self.consumers} migration_notes={self.notes}"
        )


# ---------------------------------------------------------------------------
# Registry and tree helpers
# ---------------------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataError(message)


def resolve_under(root: Path, relative: str) -> Path | None:
    """The absolute path of a repo-relative entry, or None when it escapes root."""
    if relative == "" or Path(relative).is_absolute():
        return None
    try:
        base = root.resolve(strict=False)
        resolved = (base / relative).resolve(strict=False)
    except OSError:
        return None
    if resolved == base or base in resolved.parents:
        return resolved
    return None


def load_registry(root: Path, registry_rel: str) -> list[Removal]:
    """Every ``removed<TAB>successor`` row, in file order.

    Structure is strict: a row that is not exactly one tab-separated pair of
    nonempty repo-relative paths, or a second row for a path already recorded,
    is a data error (D5, exit 2). Blank lines are skipped so the table can carry
    section spacing. An absent table is a ``DataError`` here; ``scan`` checks it
    first and reports it as rule ``S``.
    """
    table = root / registry_rel
    if not table.is_file():
        raise DataError(f"migration table absent: {registry_rel}")
    seen: dict[str, int] = {}
    rows: list[Removal] = []
    for number, raw in enumerate(table.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split("\t")
        _require(
            len(parts) == 2,
            f"D5 {registry_rel}:{number}: a row is one tab-separated removed/successor "
            f"pair, got {len(parts)} field(s): {raw!r}",
        )
        removed, successor = parts[0].strip(), parts[1].strip()
        _require(
            removed != "",
            f"D5 {registry_rel}:{number}: the removed path is empty: {raw!r}",
        )
        # An empty successor is structure-legal and refused by D1; a nonempty one
        # must be a repo-relative path inside the scanned root (D5).
        for field, value in (("removed", removed), ("successor", successor)):
            _require(
                value == "" or resolve_under(root, value) is not None,
                f"D5 {registry_rel}:{number}: the {field} path {value!r} is not a "
                "repo-relative path inside the scanned root",
            )
        if removed in seen:
            raise DataError(
                f"D5 {registry_rel}:{number}: duplicate removed path {removed!r} "
                f"(already recorded on line {seen[removed]})"
            )
        seen[removed] = number
        rows.append(Removal(removed=removed, successor=successor, line=number))
    _require(rows != [], f"registry records no removal: {registry_rel}")
    return rows


def doc_surface(root: Path) -> list[str]:
    """Every ``*.md`` file in the tree outside the excluded directories."""
    found: list[str] = []
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in EXCLUDED_DIRS and not name.startswith(".lake")
        )
        for name in sorted(names):
            if not name.endswith(DOC_SUFFIX):
                continue
            relative = (Path(current) / name).relative_to(root).as_posix()
            found.append(relative)
    return sorted(found)


def _reference_map(
    root: Path,
    needles: Sequence[str],
    documents: Sequence[str],
    successor_of: Callable[[str], str],
) -> tuple[dict[str, list[tuple[str, int]]], int]:
    """``{removed path: [(consumer, line)]}`` over the surface, plus the note count.

    One compiled alternation per scan, so the surface is read once. There is
    exactly one exemption: a reference on a line that **also names the row's
    successor** ("docs/old/intake.md is now docs/intent.md") is a migration note,
    recorded and not refused. Naming both paths is the checkable way to record a
    preimage, so the rule can accept a migration record without a heuristic.

    There is deliberately **no** exemption for a reference from the successor
    document itself. It looks safe and is not: on this tree the only two
    references in the whole surface are the raw-log table in
    `docs/evidence/perf-ledger.md`, which is *itself* the successor of the two
    `perf-global-baseline` logs it still presents as available artefacts with
    digests. A successor-self exemption would hide exactly the stale consumer this
    rule exists to find, so the successor must name the successor path beside the
    preimage to be accepted.
    """
    hits: dict[str, list[tuple[str, int]]] = {}
    notes = 0
    if not needles:
        return hits, notes
    pattern = re.compile("|".join(re.escape(needle) for needle in needles))
    for relative in documents:
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text:
            continue
        starts: list[int] = []
        offset = 0
        for line in text.splitlines(keepends=True):
            starts.append(offset)
            offset += len(line)
        for match in pattern.finditer(text):
            at = match.start()
            needle = match.group(0)
            if not _is_whole_token(text, at, len(needle)):
                continue
            successor = successor_of(needle)
            start = text.rfind("\n", 0, at) + 1
            end = text.find("\n", at)
            line_text = text[start:] if end < 0 else text[start:end]
            # A self-successor row (D2) must not exempt the references to the
            # removed path: the reference itself would satisfy the successor test.
            if successor and successor != needle and _whole_token(line_text, successor):
                notes += 1
                continue
            line_no = bisect.bisect_right(starts, start)
            hits.setdefault(needle, []).append((relative, line_no))
    return hits, notes


def _whole_token(line: str, needle: str) -> bool:
    start = 0
    while True:
        at = line.find(needle, start)
        if at < 0:
            return False
        if _is_whole_token(line, at, len(needle)):
            return True
        start = at + 1


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------

def scan(root: Path, registry_rel: str = REGISTRY) -> tuple[list[Violation], Summary]:
    """Every violation of the removal-reachability condition in the tree."""
    violations: list[Violation] = []
    if not (root / DOCS_ROOT).is_dir():
        violations.append(
            Violation(
                "S",
                DOCS_ROOT,
                0,
                f"documentation root absent: {DOCS_ROOT}/ (the migration table and the "
                "documentation surface it governs are both under it)",
            )
        )
        return violations, Summary(0, 0, 0, 0, 0, 0)
    if not (root / registry_rel).is_file():
        violations.append(
            Violation(
                "S",
                registry_rel,
                0,
                f"migration table absent: {registry_rel} (every documentation removal must "
                "name its successor here, so an absent table admits every removal unchecked)",
            )
        )
        return violations, Summary(0, 0, 0, 0, 0, 0)

    rows = load_registry(root, registry_rel)
    documents = doc_surface(root)
    successor_of = {row.removed: row.successor for row in rows}
    hits, notes = _reference_map(
        root, sorted(successor_of), documents, successor_of.get
    )

    consumers = 0
    gated = 0
    non_markdown = 0
    for row in rows:
        if row.removed.endswith(DOC_SUFFIX):
            gated += 1
        else:
            non_markdown += 1

        if row.successor == "":
            violations.append(
                Violation(
                    "D1",
                    registry_rel,
                    row.line,
                    f"removal of {row.removed} names no replacement: a documentation removal "
                    "is a migration or an authorised retirement, and both name what takes over "
                    "(AGENTS.md section 10, 'Consolidation lost files')",
                )
            )
        elif row.successor != "":
            resolved = resolve_under(root, row.successor)
            if resolved is None:
                violations.append(
                    Violation(
                        "D2",
                        registry_rel,
                        row.line,
                        f"replacement {row.successor} for {row.removed} resolves outside the "
                        "scanned root",
                    )
                )
            elif row.successor == row.removed:
                violations.append(
                    Violation(
                        "D2",
                        registry_rel,
                        row.line,
                        f"replacement for {row.removed} is the removed path itself "
                        f"({row.successor}): a removal names what takes over",
                    )
                )
            elif not resolved.is_file():
                violations.append(
                    Violation(
                        "D2",
                        registry_rel,
                        row.line,
                        f"replacement {row.successor} for {row.removed} does not exist under "
                        "the scanned root",
                    )
                )

        removed_path = resolve_under(root, row.removed)
        if removed_path is not None and removed_path.exists():
            violations.append(
                Violation(
                    "D3",
                    row.removed,
                    0,
                    f"registered as removed on {registry_rel}:{row.line} but still present in "
                    "the tree, so the migration table is stale in the other direction",
                )
            )

        found = hits.get(row.removed, [])
        if found:
            consumers += len(found)
            by_document: dict[str, list[int]] = {}
            for consumer, line_no in found:
                by_document.setdefault(consumer, []).append(line_no)
            for consumer in sorted(by_document):
                lines = sorted(by_document[consumer])
                shown = ",".join(str(number) for number in lines)
                successor = f"; its named successor is {row.successor}" if row.successor else ""
                violations.append(
                    Violation(
                        "D4",
                        consumer,
                        lines[0],
                        f"live document still names the removed path {row.removed} at "
                        f"line(s) {shown}{successor}",
                    )
                )

    summary = Summary(
        rows=len(rows),
        documents=len(documents),
        gated=gated,
        non_markdown=non_markdown,
        consumers=consumers,
        notes=notes,
    )
    return violations, summary


# ---------------------------------------------------------------------------
# --check, --list, --inventory
# ---------------------------------------------------------------------------

def check(root: Path, registry_rel: str) -> int:
    violations, summary = scan(root, registry_rel)
    if not violations:
        print(
            f"CHECK CLEAN: {registry_rel} records {summary.rows} removal(s); every "
            "replacement resolves, every removed path is absent and no document names one"
        )
        print(summary.render())
        return EXIT_OK
    for violation in violations:
        print(violation.render())
    print(summary.render())
    by_rule: dict[str, int] = {}
    for violation in violations:
        by_rule[violation.rule] = by_rule.get(violation.rule, 0) + 1
    detail = ", ".join(f"{rule}={count}" for rule, count in sorted(by_rule.items()))
    print(f"CHECK FAIL: {len(violations)} violation(s) ({detail})")
    return EXIT_REJECT


def inventory(root: Path, registry_rel: str, removed: str) -> int:
    """The consumer inventory of one registered removal. Non-empty is a refusal."""
    rows = load_registry(root, registry_rel)
    row = next((entry for entry in rows if entry.removed == removed), None)
    if row is None:
        raise DataError(
            f"{registry_rel} records no removal of {removed}: the consumer inventory is "
            "only defined for a registered removal"
        )
    documents = doc_surface(root)
    hits, _notes = _reference_map(root, [removed], documents, {removed: row.successor}.get)
    found = sorted(hits.get(removed, []))
    print(f"INVENTORY {removed} (registry line {row.line}, successor {row.successor or 'none'})")
    for consumer, line_no in found:
        print(f"  {consumer}:{line_no}")
    print(f"  consumers={len(found)} of {len(documents)} document(s) scanned")
    if found:
        print(
            f"INVENTORY FAIL: {len(found)} live document(s) still name {removed} "
            "(rule D4)"
        )
        return EXIT_REJECT
    print("INVENTORY CLEAN: no document in the surface names the removed path")
    return EXIT_OK


def cmd_list(root: Path, registry_rel: str) -> int:
    print(f"Registry: {registry_rel} (one 'removed<TAB>successor' row per removal)")
    print(f"Surface: every *{DOC_SUFFIX} file outside "
          f"{', '.join(sorted(EXCLUDED_DIRS))} and *.lake")
    print("Exemption (recorded as a migration note, never refused): a reference on a line "
          "that also names the row's successor; a reference from the successor itself is "
          "NOT exempt")
    print("Rules:")
    for rule, title in (
        ("S", "the registry or the documentation root is absent"),
        ("D1", "a row names no replacement"),
        ("D2", "a replacement resolves to no file under the root, or names the removed path"),
        ("D3", "a registered removed path is still present in the tree"),
        ("D4", "a document in the surface names a removed path (the reachability check)"),
        ("D5", "a malformed, empty, escaping or duplicate registry row (data error, exit 2)"),
    ):
        print(f"  {rule}: {title}")
    table = root / registry_rel
    if not table.is_file():
        print(f"  (no registry at {registry_rel} in this tree)")
        return EXIT_OK
    rows = load_registry(root, registry_rel)
    documents = doc_surface(root)
    successor_of = {row.removed: row.successor for row in rows}
    hits, notes = _reference_map(root, sorted(successor_of), documents, successor_of.get)
    populated = {name: found for name, found in hits.items() if found}
    print(f"Rows in {registry_rel}: {len(rows)}")
    print(f"Documents in the surface: {len(documents)}")
    print(f"Removals with a non-empty consumer inventory: {len(populated)}")
    print(f"Migration notes recorded: {notes}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# --self-test: a synthetic tree, never the repository
# ---------------------------------------------------------------------------

_CLEAN_REGISTRY = (
    "docs/old/intake.md\tdocs/intent.md\n"
    "docs/old/bench.md\tdocs/evidence.md\n"
    "docs/old/trace.log\tdocs/evidence.md\n"
)

_CLEAN_DOCS = {
    "docs/intent.md": "# intent\n\nSupersedes docs/old/intake.md; the live copy is docs/intent.md.\n",
    "docs/evidence.md": "# evidence\n\nBench receipts live here.\n",
    "docs/plan.md": "# plan\n\nRead [the intent](intent.md) first.\n",
    "docs/old/README.md": "# old\n\nThis directory is gone except this note.\n",
    "README.md": "# readme\n\nSee docs/plan.md.\n",
    "tools/probe.py": '"""A tool that names a removed path in a record."""\n'
    'REMOVED = "docs/old/intake.md"\n',
    ".tracks/notes.md": "Local scratch naming docs/old/intake.md; not tree state.\n",
}

CONTROL_LINE = (
    "Seeded by tools/rooting/doc_reach.py --controls; not tree state.\n"
)


def _materialize_clean(root: Path) -> None:
    for relative, text in _CLEAN_DOCS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / REGISTRY).parent.mkdir(parents=True, exist_ok=True)
    (root / REGISTRY).write_text(_CLEAN_REGISTRY, encoding="utf-8")


def _append(root: Path, relative: str, text: str) -> None:
    with (root / relative).open("a", encoding="utf-8") as handle:
        handle.write(text)


def _replace_in_file(root: Path, relative: str, old: str, new: str) -> None:
    path = root / relative
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )


def _seed_no_successor(root: Path) -> None:
    _replace_in_file(root, REGISTRY, "docs/old/intake.md\tdocs/intent.md", "docs/old/intake.md\t")


def _seed_unreachable_successor(root: Path) -> None:
    _replace_in_file(
        root,
        REGISTRY,
        "docs/old/intake.md\tdocs/intent.md",
        "docs/old/intake.md\tdocs/nowhere/absent.md",
    )


def _seed_successor_is_removed(root: Path) -> None:
    _replace_in_file(
        root,
        REGISTRY,
        "docs/old/intake.md\tdocs/intent.md",
        "docs/old/intake.md\tdocs/old/intake.md",
    )


def _seed_live_consumer(root: Path) -> None:
    _append(root, "docs/plan.md", "\nThe old intake lives in docs/old/intake.md.\n")


def _seed_successor_names_preimage(root: Path) -> None:
    """The false negative a successor-self exemption would hide."""
    _replace_in_file(
        root,
        "docs/intent.md",
        "Supersedes docs/old/intake.md; the live copy is docs/intent.md.",
        "| `docs/old/intake.md` | `92badfedd7f756d90` | 492 |",
    )


def _seed_removed_path_present(root: Path) -> None:
    path = root / "docs/old/intake.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# resurrected\n", encoding="utf-8")


def _seed_row_without_tab(root: Path) -> None:
    _append(root, REGISTRY, "docs/old/loose.md docs/intent.md\n")


def _seed_duplicate_removed(root: Path) -> None:
    _append(root, REGISTRY, "docs/old/intake.md\tdocs/intent.md\n")


def _seed_row_with_three_fields(root: Path) -> None:
    _append(root, REGISTRY, "docs/old/loose.md\tdocs/intent.md\tdocs/plan.md\n")


def _seed_empty_removed_path(root: Path) -> None:
    _append(root, REGISTRY, "\tdocs/intent.md\n")


def _seed_empty_registry(root: Path) -> None:
    """Whitespace only: the field-count check passes, the emptiness guard must fire."""
    (root / REGISTRY).write_text("\n   \n\t\n", encoding="utf-8")


def _seed_absolute_removed_path(root: Path) -> None:
    _append(root, REGISTRY, "/etc/passwd\tdocs/intent.md\n")


def _seed_escaping_removed_path(root: Path) -> None:
    _append(root, REGISTRY, "../outside.md\tdocs/intent.md\n")


def _seed_escaping_successor(root: Path) -> None:
    _replace_in_file(
        root,
        REGISTRY,
        "docs/old/bench.md\tdocs/evidence.md",
        "docs/old/bench.md\t../outside.md",
    )


def _seed_migration_note(root: Path) -> None:
    _append(
        root,
        "docs/plan.md",
        "\nIntake moved: docs/old/intake.md is now docs/intent.md.\n",
    )


def _seed_reference_outside_surface(root: Path) -> None:
    _append(root, "tools/probe.py", 'ALSO = "docs/old/bench.md"\n')
    _append(root, ".tracks/notes.md", "Also docs/old/bench.md.\n")


def _seed_reference_inside_longer_token(root: Path) -> None:
    """A longer path that contains a removed path is not a reference to it."""
    _append(root, "docs/plan.md", "\nBackup: docs/old/intake.md.bak, copied from intake.md.\n")


def _seed_self_successor_with_reference(root: Path) -> None:
    """A self-successor row must not turn its own references into migration notes."""
    _replace_in_file(
        root,
        REGISTRY,
        "docs/old/bench.md\tdocs/evidence.md",
        "docs/old/bench.md\tdocs/old/bench.md",
    )
    _append(root, "docs/plan.md", "\nThe bench receipts are in docs/old/bench.md.\n")


def _seed_registry_absent(root: Path) -> None:
    (root / REGISTRY).unlink()


def _seed_docs_root_absent(root: Path) -> None:
    shutil.rmtree(root / DOCS_ROOT, ignore_errors=True)


def _fresh_copy(base: Path, prefix: str) -> Path:
    work = Path(tempfile.mkdtemp(prefix=prefix))
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(base, work, dirs_exist_ok=True)
    return work


def self_test() -> int:
    """Accept the clean fixture; refuse each seeded defect with its own rule."""
    failures: list[str] = []
    red: list[str] = []
    base = Path(tempfile.mkdtemp(prefix="doc-reach-clean-"))
    try:
        _materialize_clean(base)
        clean, clean_summary = scan(base)
        if clean:
            print(f"SELF-TEST FAIL clean fixture: {clean[0].render()}")
            return EXIT_REJECT
        print(
            f"SELF-TEST GREEN clean fixture ({clean_summary.rows} rows, "
            f"{clean_summary.documents} documents, {clean_summary.consumers} consumers)"
        )

        cases: list[tuple[str, str, str, Callable[[Path], None]]] = [
            ("migration table absent", "S", "migration table absent", _seed_registry_absent),
            ("documentation root absent", "S", "documentation root absent", _seed_docs_root_absent),
            ("no replacement named", "D1", "names no replacement", _seed_no_successor),
            (
                "unreachable replacement",
                "D2",
                "does not exist under the scanned root",
                _seed_unreachable_successor,
            ),
            (
                "replacement is the removed path",
                "D2",
                "is the removed path itself",
                _seed_successor_is_removed,
            ),
            (
                "live consumer still names the removed document",
                "D4",
                "live document still names the removed path docs/old/intake.md",
                _seed_live_consumer,
            ),
            (
                "successor document still names its preimage",
                "D4",
                "live document still names the removed path docs/old/intake.md",
                _seed_successor_names_preimage,
            ),
            (
                "self-successor row does not exempt its references",
                "D4",
                "live document still names the removed path docs/old/bench.md",
                _seed_self_successor_with_reference,
            ),
            (
                "registered removal still on disk",
                "D3",
                "still present in the tree",
                _seed_removed_path_present,
            ),
            ("row is not a tab-separated pair", "D5", "tab-separated", _seed_row_without_tab),
            ("row carries three fields", "D5", "tab-separated", _seed_row_with_three_fields),
            ("removed path is empty", "D5", "removed path is empty", _seed_empty_removed_path),
            ("duplicate removed path", "D5", "duplicate removed path", _seed_duplicate_removed),
            ("removed path is absolute", "D5", "repo-relative path inside", _seed_absolute_removed_path),
            ("removed path escapes the root", "D5", "repo-relative path inside", _seed_escaping_removed_path),
            ("successor escapes the root", "D5", "repo-relative path inside", _seed_escaping_successor),
            ("registry records no removal", "D5", "records no removal", _seed_empty_registry),
            ("migration note naming the successor", "-", "", _seed_migration_note),
            ("reference outside the documentation surface", "-", "", _seed_reference_outside_surface),
            ("removed path inside a longer token", "-", "", _seed_reference_inside_longer_token),
        ]
        for name, rule, needle, seed in cases:
            work = _fresh_copy(base, "doc-reach-case-")
            try:
                seed(work)
                try:
                    violations, _summary = scan(work)
                except DataError as exc:
                    if rule == "D5":
                        print(f"SELF-TEST RED  {name}: data error: {exc}")
                        red.append(f"{rule} {name}")
                        continue
                    failures.append(f"{name}: data error where rule {rule} was expected: {exc}")
                    print(f"SELF-TEST FAIL {name}: data error: {exc}")
                    continue
                if rule == "-":
                    if violations:
                        failures.append(f"{name}: rejected, expected acceptance")
                        print(f"SELF-TEST FAIL {name}: {violations[0].render()}")
                    else:
                        print(f"SELF-TEST GREEN {name}")
                    continue
                hit = next((v for v in violations if v.rule == rule), None)
                if hit is None:
                    found = sorted({v.rule for v in violations}) or "no rule"
                    failures.append(f"{name}: raised {found}, expected {rule}")
                    print(f"SELF-TEST FAIL {name}: raised {found}, expected {rule}")
                    continue
                if needle and needle not in hit.message:
                    failures.append(f"{name}: rejected without the expected phrase {needle!r}")
                    print(f"SELF-TEST FAIL {name}: {hit.render()}")
                    continue
                print(f"SELF-TEST RED  {name}: {hit.render()}")
                red.append(f"{rule} {name}")
            finally:
                shutil.rmtree(work, ignore_errors=True)

        # The inventory leg: the same removal, printed rather than scanned.
        work = _fresh_copy(base, "doc-reach-inventory-")
        try:
            if inventory(work, REGISTRY, "docs/old/intake.md") != EXIT_OK:
                failures.append("inventory: clean fixture refused")
                print("SELF-TEST FAIL inventory: clean fixture refused")
            _seed_live_consumer(work)
            if inventory(work, REGISTRY, "docs/old/intake.md") != EXIT_REJECT:
                failures.append("inventory: live consumer accepted")
                print("SELF-TEST FAIL inventory: live consumer accepted")
            else:
                print("SELF-TEST RED  inventory of a removed path: non-empty inventory is refused")
                red.append("D4 consumer inventory")
        finally:
            shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if failures:
        print(f"SELF-TEST FAIL: {len(failures)} case(s) failed")
        return EXIT_REJECT
    rejects = len(red)
    accepts = 1 + sum(1 for name, rule, _n, _s in cases if rule == "-")
    print(
        f"SELF-TEST PASS: clean fixture accepted, {rejects} rejection fixture(s) fired "
        f"their rule, {accepts} accept fixture(s) green"
    )
    print(f"SELF-TEST PASS: rejected: {', '.join(red)}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# --controls: baseline / mutation / restore on a copy of the real docs subtree
# ---------------------------------------------------------------------------

def _docs_copy(root: Path, prefix: str) -> Path:
    work = Path(tempfile.mkdtemp(prefix=prefix))
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(root / DOCS_ROOT, work / DOCS_ROOT, dirs_exist_ok=True)
    return work


def _verdict(violations: Iterable[Violation]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    found = [violation.render() for violation in violations]
    return tuple(sorted(v.rule for v in violations)), tuple(sorted(found))


def controls(root: Path, registry_rel: str) -> int:
    """Baseline, then each leg red under a seeded defect, then restore.

    The real ``docs/`` subtree is copied into the system temporary directory and
    mutated there; the working tree is read-only. The baseline verdict is
    reported as found, not required to be green: a pre-existing violation in the
    tree is a finding for the lead, and the mutation legs must fire with a rule
    that is not already in the baseline.
    """
    base = _docs_copy(root, "doc-reach-base-")
    failures: list[str] = []
    work: Path | None = None
    try:
        baseline, base_summary = scan(base, registry_rel)
        base_rules, base_lines = _verdict(baseline)
        state = "GREEN" if not baseline else "RED"
        print(
            f"CONTROL baseline: {state} ({base_summary.rows} removal(s), "
            f"{base_summary.documents} document(s), {base_summary.consumers} consumer ref)"
        )
        for line in base_lines:
            print(f"  pre-existing: {line}")

        rows = load_registry(base, registry_rel)
        needle = rows[0].removed

        # Leg 1 — seed a reference to a registered removal; the reachability check
        # must refuse it by name, with a diagnostic the baseline did not carry.
        work = _fresh_copy(base, "doc-reach-mutant-")
        try:
            _append(
                work,
                f"{DOCS_ROOT}/README.md",
                f"\n{CONTROL_LINE.strip()} names {needle}\n",
            )
            seeded, _ = scan(work, registry_rel)
            hit = next(
                (
                    v
                    for v in seeded
                    if v.rule == "D4" and needle in v.message and v.render() not in base_lines
                ),
                None,
            )
            if hit is None:
                found = sorted({v.rule for v in seeded}) or "no rule"
                failures.append(f"seed-reference: no attributable D4 for {needle}")
                print(f"CONTROL seed-reference: FAIL :: {found}, expected D4 for {needle}")
            else:
                print(f"CONTROL seed-reference: RED :: {hit.render()}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
            work = None

        # Leg 2 — remove a document that a live document still names, and record
        # the removal: the reachability check must refuse by name.
        pair = _find_live_pair(base, registry_rel)
        if pair is None:
            failures.append("remove-referenced: no live (consumer, document) pair to remove")
            print("CONTROL remove-referenced: FAIL :: the copy carries no referenced document")
        else:
            consumer, victim = pair
            successor = "docs/plan.md" if victim != "docs/plan.md" else "docs/README.md"
            if not (base / successor).is_file():
                failures.append(f"remove-referenced: no successor document {successor} in the copy")
                print(f"CONTROL remove-referenced: FAIL :: {successor} absent from the copy")
            else:
                work = _fresh_copy(base, "doc-reach-mutant-")
                try:
                    (work / victim).unlink()
                    _append(work, registry_rel, f"{victim}\t{successor}\n")
                    seeded, _ = scan(work, registry_rel)
                    hit = next(
                        (
                            v
                            for v in seeded
                            if v.rule == "D4"
                            and victim in v.message
                            and consumer in v.path
                            and v.render() not in base_lines
                        ),
                        None,
                    )
                    if hit is None:
                        found = sorted({v.rule for v in seeded}) or "no rule"
                        failures.append(f"remove-referenced: no attributable D4 for {victim}")
                        print(
                            f"CONTROL remove-referenced: FAIL :: {found}, expected D4 naming "
                            f"{victim} at {consumer}"
                        )
                    else:
                        print(f"CONTROL remove-referenced: RED :: {hit.render()}")
                        print(
                            f"  (removed {victim}, recorded {victim} -> {successor}, still named "
                            f"by {consumer})"
                        )
                finally:
                    shutil.rmtree(work, ignore_errors=True)
                    work = None

        # Leg 3 — restore: a pristine copy must reproduce the baseline verdict.
        work = _fresh_copy(base, "doc-reach-restore-")
        try:
            restored, _ = scan(work, registry_rel)
            if _verdict(restored) == (base_rules, base_lines):
                print(f"CONTROL restore: {state} (byte-identical to baseline)")
            else:
                failures.append("restore: a pristine copy does not reproduce the baseline")
                print("CONTROL restore: FAIL :: verdict differs from baseline")
        finally:
            shutil.rmtree(work, ignore_errors=True)
            work = None

        # Leg 4 — repair: rewrite each pre-existing consumer to name the successor.
        # Not exercised when the baseline carries no violation, and reported as
        # skipped rather than passed in that case.
        if baseline:
            work = _fresh_copy(base, "doc-reach-mutant-")
            try:
                rows_by_removed = {row.removed: row for row in rows}
                for violation in baseline:
                    if violation.rule != "D4":
                        continue
                    removed = _removed_named(violation.message, rows_by_removed)
                    if removed is None:
                        failures.append(f"repair: no registered removal named by {violation.render()}")
                        continue
                    _replace_in_file(
                        work,
                        violation.path,
                        removed,
                        rows_by_removed[removed].successor,
                    )
                repaired, repaired_summary = scan(work, registry_rel)
                if repaired:
                    failures.append("repair: rewriting the pre-existing consumers is not green")
                    print(f"CONTROL repair: FAIL :: {repaired[0].render()}")
                else:
                    print(
                        f"CONTROL repair: GREEN ({repaired_summary.rows} removal(s), "
                        f"{repaired_summary.consumers} consumer ref, 0 violation)"
                    )
            finally:
                shutil.rmtree(work, ignore_errors=True)
                work = None
        else:
            print("CONTROL repair: SKIPPED (the baseline carries no violation to repair)")
    finally:
        shutil.rmtree(base, ignore_errors=True)
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)

    if failures:
        print(f"CONTROLS FAIL: {len(failures)} leg(s) failed", file=sys.stderr)
        return EXIT_REJECT
    print("CONTROLS PASS: baseline reported, every seeded defect red by name, restore reproduces it")
    return EXIT_OK


def _removed_named(message: str, rows: dict[str, Removal]) -> str | None:
    for removed in rows:
        if removed in message:
            return removed
    return None


def _find_live_pair(root: Path, registry_rel: str) -> tuple[str, str] | None:
    """A (consumer document, referenced document) pair inside the copy."""
    documents = doc_surface(root)
    existing = [
        relative
        for relative in documents
        if relative != registry_rel and (root / relative).is_file()
    ]
    hits, _notes = _reference_map(
        root, existing, documents, lambda _name: ""
    )
    for victim in sorted(hits):
        if victim.startswith(f"{DOCS_ROOT}/old/"):
            continue
        for consumer, _line in sorted(hits[victim]):
            if consumer != victim:
                return consumer, victim
    return None


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="doc_reach.py",
        description=(
            "Reachability gate for documentation removals: every removal registered in "
            f"{REGISTRY} names a resolvable successor, is absent from the tree and is "
            "named by no live document (S, D1-D5)."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="print the surface, the rules and the row count")
    mode.add_argument("--self-test", action="store_true", help="exercise the seeded-defect fixtures")
    mode.add_argument(
        "--controls",
        action="store_true",
        help="baseline/mutation/restore against a copy of the real docs subtree",
    )
    mode.add_argument("--check", action="store_true", help="scan the repository tree")
    mode.add_argument(
        "--inventory",
        metavar="PATH",
        help="print the consumer inventory of one registered removal (exit 1 if non-empty)",
    )
    parser.add_argument("--root", default=str(REPO), metavar="PATH", help="tree to scan")
    parser.add_argument("--registry", default=REGISTRY, metavar="PATH", help="migration table to read")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    registry = args.registry
    try:
        if args.self_test:
            return self_test()
        if args.list:
            return cmd_list(root, registry)
        if args.inventory:
            return inventory(root, registry, args.inventory)
        if args.controls:
            return controls(root, registry)
        return check(root, registry)
    except GateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())