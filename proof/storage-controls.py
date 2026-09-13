#!/usr/bin/env python3
"""Durable mutation-control runner for the fixed-shard storage refinement epic.

Each control applies one documented mutation to a throwaway copy of the proof
tree, runs the **whole-package** ``lake build`` (the ``ApeironWordBridge`` default
target, not ``lake build <module>``) in the copy, and asserts that the build fails
at the recorded **declaration**: the first located ``error:`` names a module, and
the enclosing top-level ``theorem``/``def``/``lemma`` at that position must equal
the control's expected identifier. It then restores the pristine source and
asserts that the same whole-package build succeeds. A control that does not go red
then green is a failure, and the process exits nonzero.

Why names, not lines
--------------------
A line number is a property of the module at one revision. The generic admission
material moved between ``TableRefinementAdmissionGeneric.lean`` and
``TableRefinementReachablePrefixExactN.lean`` without changing any theorem, and any
future edit shifts every later line. An identifier plus its module survives both.
The observed ``file:line:col`` is still parsed, reported and compared to the
historical recorded locator, but the assertion is on the resolved declaration name.
The name is not read from the Lean message (which does not carry it); it is resolved
from the mutated copy by scanning the error file for the last top-level declaration
header at or before the error line.

Why the whole package
--------------------
The recorded failure locators are properties of the package default target. A
module-scoped build can report a different first error, or none, depending on
which modules the cache already holds. Building the default target makes the
control independent of a concurrent owner's module edits and reproduces the
locator the plan records. It also means the first error is the first error a
clean package build reports, not the first error of one chosen module.

Why a copy at all
-----------------
The runner must never mutate the repository working tree and must not share the
Lean build lane owned by a concurrent task. It therefore builds only inside a
throwaway directory. The copy method is chosen so that a write inside the copy
cannot reach the original:

* sources and the small ``.lake/build`` / ``.lake/config`` trees are real copies;
* the large, read-only dependency cache (``.lake/packages``: Mathlib, aesop,
  batteries, the pinned corpus, ...) is referenced through a symlink. Lake sees
  the original absolute paths, so the cache replays and is never rebuilt;
* the mutated source in the copy is verified to be a regular file, and the
  original generated source, the mutated original, the repository
  ``git status --porcelain`` output and the package caches are checked for writes
  after every build.

Usage
-----
::

    python3 proof/storage-controls.py --list
    python3 proof/storage-controls.py --self-test
    python3 proof/storage-controls.py --only w11-t5
    python3 proof/storage-controls.py --only w11-t5 --only s6-fresh-write
    python3 proof/storage-controls.py --jobs 2
    python3 proof/storage-controls.py --throwaway-root /tmp/opencode/storage-controls

``--list`` prints every encoded control with its module, expected declaration and
informational historical locator, plus each decided control and the exact reason
it has no faithful encoding. ``--self-test`` mutates the runner's own expected
name and confirms the red assertion fires, proves a surviving mutation and a wrong
name are rejected, proves every encoded mutation resolves against the real source,
proves the decided control is rejected, and proves the missing-tool gate rejects
an absent ``lake`` with exit 3. It needs no build. Every control prints one JSON
object; the process exits 0 only when every selected control goes red then green.

Mutation safety
---------------
Mutations are exact string replacements. ``exact`` requires exactly one
occurrence of the target text. ``scoped`` requires exactly one anchor and at
least one target after it, and replaces the first target after the anchor; this
is how the three identical generated loop bodies are distinguished by their
``def`` header. A missing target, a missing anchor or an ambiguous anchor is a
hard error, never a silent no-op.

Exit status
-----------
* 0 - every selected control went red then green;
* 1 - at least one control failed, or the repository changed under the runner;
* 2 - usage error;
* 3 - required tool missing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_FILE = "proof/ApeironWord.lean"
PACKAGE_DIRS = (
    "proof/.lake/packages",
    "proof/corpus/.lake/packages",
)
DEFAULT_THROWAWAY_ROOT = Path("/tmp/opencode/storage-controls")
DEFAULT_TIMEOUT = 1800

ERROR_RE = re.compile(r"^error:\s+(\S+?):(\d+):(\d+):", re.MULTILINE)

# A top-level Lean declaration header, at column 0. The name is the identifier the
# error position resolves to. Modifiers are skipped; attributes and doc comments sit
# on their own lines and do not need to match.
DECL_RE = re.compile(
    r"^(?:private\s+|protected\s+|noncomputable\s+|partial\s+|unsafe\s+)*"
    r"(?:theorem|lemma|def|abbrev|structure|class|inductive|instance)\s+"
    r"([A-Za-z_][A-Za-z0-9_.'!?]*)"
)


class ControlError(RuntimeError):
    """A control's own setup, mutation or assertion failed."""


class MutationError(ControlError):
    """The documented mutation target was absent or ambiguous."""


class ToolMissingError(ControlError):
    """A required executable was not found on ``PATH``."""


# ---------------------------------------------------------------------------
# Control definitions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Expect:
    """The module and top-level declaration a control's red build must name."""

    module: str
    name: str

    def __str__(self) -> str:
        return f"{self.module}/{self.name}"


@dataclasses.dataclass(frozen=True)
class Mutation:
    """An exact string replacement, optionally scoped to a unique anchor."""

    new: str
    old: str
    anchor: Optional[str] = None

    def apply(self, text: str) -> str:
        if self.anchor is None:
            count = text.count(self.old)
            if count != 1:
                raise MutationError(
                    f"exact target occurs {count} times, expected exactly 1: "
                    f"{self.old!r}"
                )
            return text.replace(self.old, self.new, 1)

        anchor_count = text.count(self.anchor)
        if anchor_count != 1:
            raise MutationError(
                f"anchor occurs {anchor_count} times, expected exactly 1: "
                f"{self.anchor!r}"
            )
        start = text.index(self.anchor)
        target_at = text.find(self.old, start)
        if target_at < 0:
            raise MutationError(
                f"scoped target absent after anchor {self.anchor!r}: {self.old!r}"
            )
        return text[:target_at] + self.new + text[target_at + len(self.old):]


@dataclasses.dataclass(frozen=True)
class Control:
    """One encoded documented mutation control."""

    id: str
    summary: str
    mutation: Mutation
    source_file: str
    expect: Expect
    recorded: str = ""
    note: str = ""


@dataclasses.dataclass(frozen=True)
class Decided:
    """A documented control that has no faithful encoded mutation.

    ``decided`` is not ``unencoded``: the runner rejects the requested id with
    exit 2 and prints ``reason``. The decision and its reason are part of the
    control table, so a future reader knows the omission was considered.
    """

    id: str
    summary: str
    reason: str


def _loop0(anchor: str, old: str, new: str) -> Mutation:
    return Mutation(anchor=anchor, old=old, new=new)


# The generated `insert_with` has three byte-identical loop bodies. A `scoped`
# mutation anchors on the enclosing `def` header so only the intended body is
# changed.
_LOOP0 = "def table.Table.insert_with_loop0.body"
_INSERT_WITH = "def table.Table.insert_with\n"

# The expected declaration below is the top-level `theorem`/`def` enclosing the
# first located error of the whole-package `lake build` under the mutation,
# observed in a throwaway copy at the source revision this runner ships with. The
# `recorded` field is the historical `file:line:col`; it is informational only and
# a future shift is printed as `(drift)` rather than failing the control.
CONTROLS: tuple[Control, ...] = (
    Control(
        id="w11-t1",
        summary="generated present-key branch stores `value` instead of `merge value old`",
        mutation=Mutation(
            old="        let slots1 := index_mut_back1 (some (t, stored1))",
            new="        let slots1 := index_mut_back1 (some (t, value))",
        ),
        source_file=GENERATED_FILE,
        expect=Expect("TableRefinement.lean", "table_insert_present_zero"),
        recorded="TableRefinement.lean:203:79",
    ),
    Control(
        id="w11-t2",
        summary="generated get present branch returns `self.default`",
        mutation=Mutation(
            old="                  ok value\n",
            new="                  ok self.default\n",
        ),
        source_file=GENERATED_FILE,
        expect=Expect("TableRefinement.lean", "word_table_get_new_first"),
        recorded="TableRefinement.lean:174:88",
    ),
    Control(
        id="w11-t3",
        summary=(
            "generated insert_with_loop0 vacant branch stores `value` instead of "
            "`wrapping_add value default`"
        ),
        mutation=_loop0(
            _LOOP0,
            "Array.update slots1 i (some (key, merged))",
            "Array.update slots1 i (some (key, value))",
        ),
        source_file=GENERATED_FILE,
        expect=Expect(
            "TableRefinementGeneric.lean", "insert_with_loop0_vacant_body_exact"
        ),
        recorded="TableRefinementGeneric.lean:1041:69",
    ),
    Control(
        id="w11-t5",
        summary="generated insert_with_loop0 vacant branch guard `if b` -> `if false`",
        mutation=_loop0(_LOOP0, "    if b\n    then", "    if false\n    then"),
        source_file=GENERATED_FILE,
        expect=Expect(
            "TableRefinementGeneric.lean", "insert_with_loop0_vacant_body_exact"
        ),
        recorded="TableRefinementGeneric.lean:1062:17",
    ),
    Control(
        id="w12-t1",
        summary="generated build_loop guard `i < i1` -> `i.val + 1 < i1.val`",
        mutation=Mutation(old="  if i < i1\n", new="  if i.val + 1 < i1.val\n"),
        source_file=GENERATED_FILE,
        expect=Expect("TableRefinement.lean", "table_build_loop_two"),
        recorded="TableRefinement.lean:297:8",
    ),
    Control(
        id="w12-t2",
        summary=(
            "generated insert_with_loop0 vacant branch writes at `i + 1` instead "
            "of the first vacant slot `i`"
        ),
        mutation=_loop0(
            _LOOP0,
            "      let slots1 := index_mut_back1 o1\n"
            "      let slots2 \u2190 Array.update slots1 i (some (key, merged))",
            "      let slots1 := index_mut_back1 o1\n"
            "      let i9 \u2190 i + 1#usize\n"
            "      let slots2 \u2190 Array.update slots1 i9 (some (key, merged))",
        ),
        source_file=GENERATED_FILE,
        expect=Expect(
            "TableRefinementGeneric.lean", "insert_with_loop0_vacant_body_exact"
        ),
        recorded="TableRefinementGeneric.lean:1041:69",
        note=(
            "plan recorded `tableInv_insert_with`; the whole-package build now "
            "fails first in the generic vacant-body exactness theorem"
        ),
    ),
    Control(
        id="w12-t3",
        summary=(
            "generated find_slots_loop reports `done none` for a present key "
            "(present-key-as-absent)"
        ),
        mutation=Mutation(old="then ok (done (some i))", new="then ok (done none)"),
        source_file=GENERATED_FILE,
        expect=Expect("TableRefinement.lean", "oneSlots_find"),
        recorded="TableRefinement.lean:153:4",
        note=(
            "plan recorded the pre-split present-key theorem; the whole-package "
            "build now fails first in the pre-existing oneSlots_find present-read "
            "theorem"
        ),
    ),
    Control(
        id="w12-t4",
        summary=(
            "proof-level duplicate-on-present: `replacePresent` (the model of the "
            "generated present-branch update) also writes the entry at slot 0, so a "
            "located present key at `i != 0` is stored twice"
        ),
        mutation=Mutation(
            old="some (slots.set i (some (storedKey, value)))",
            new=(
                "some ((slots.set i (some (storedKey, value))).set 0#usize "
                "(some (storedKey, value)))"
            ),
        ),
        source_file="proof/corpus/TableRefinementGeneric.lean",
        expect=Expect("TableRefinementGeneric.lean", "insert_with_present_exact"),
        recorded="TableRefinementGeneric.lean:60:50",
        note=(
            "the documented generated-body mutation (present branch writes at the "
            "first vacant slot) needs branch restructuring; this encodes the same "
            "duplicate-on-present defect at the proof level by mutating the model "
            "definition `replacePresent`, and the present-branch refinement lemma "
            "rejects it"
        ),
    ),
    Control(
        id="w12-t6",
        summary=(
            "generated insert_with_loop0 full-shard branch returns `Ok ()` with "
            "unchanged state instead of `Err ShardFull`"
        ),
        mutation=_loop0(
            _LOOP0,
            "ok (done (core.result.Result.Err table.StorageError.ShardFull, a))",
            "ok (done (core.result.Result.Ok (), a))",
        ),
        source_file=GENERATED_FILE,
        expect=Expect("TableRefinement.lean", "insert_full_loop_spec"),
        recorded="TableRefinement.lean:517:4",
    ),
    Control(
        id="s6-fresh-write",
        summary=(
            "generated fresh-write call `merge value t` -> `merge t value` "
            "(README table-contract control)"
        ),
        mutation=_loop0(
            _INSERT_WITH,
            "MergeInst.merge merge value t\n",
            "MergeInst.merge merge t value\n",
        ),
        source_file=GENERATED_FILE,
        expect=Expect("TableRefinement.lean", "table_insert_empty_zero"),
        recorded="TableRefinement.lean:107:67",
    ),
    Control(
        id="w14-admission-generic",
        summary=(
            "proof-level admission split: `BuildAdmissibleN`'s capacity disjunct "
            "`card < N.val` weakened to `card < N.val + 1`"
        ),
        mutation=Mutation(
            anchor=(
                "def BuildAdmissibleN (N : Usize) (_default : I64) "
                "(kvs : Slice (I64 × I64)) : Prop :="
            ),
            old="(routedStreamKeys kvs i.val shard).card < N.val",
            new="(routedStreamKeys kvs i.val shard).card < N.val + 1",
        ),
        source_file="proof/corpus/TableRefinementAdmissionGeneric.lean",
        expect=Expect(
            "TableRefinementAdmissionGeneric.lean",
            "buildAdmissibleN_of_distinctKeysPerShardN",
        ),
        recorded="TableRefinementAdmissionGeneric.lean:32:15",
        note=(
            "red line lands inside the post-split admission module, not upstream, "
            "so the generic admission headline is falsified in its own file"
        ),
    ),
    Control(
        id="w14-reachable-exact",
        summary=(
            "proof-level reachable-prefix split: `ReachablePrefixExactN`'s locator "
            "clause weakened with `∨ True`"
        ),
        mutation=Mutation(
            old=(
                "(∃ slot, locate key = some (shard, slot)) ↔ "
                "key ∈ routedStreamKeys kvs i shard"
            ),
            new=(
                "(∃ slot, locate key = some (shard, slot)) ↔ "
                "(key ∈ routedStreamKeys kvs i shard ∨ True)"
            ),
        ),
        source_file="proof/corpus/TableRefinementReachablePrefixExactN.lean",
        expect=Expect(
            "TableRefinementReachablePrefixExactN.lean",
            "emptyWordTableN_reachablePrefixExactN",
        ),
        recorded="TableRefinementReachablePrefixExactN.lean:799:98",
        note=(
            "red line lands inside the post-split reachable-prefix module, so the "
            "exact-coverage headline is falsified in its own file"
        ),
    ),
)

# Documented plan controls with no faithful encoded mutation. Each is a decision
# with a recorded reason, and the runner rejects the id with exit 2 rather than
# silently dropping it.
DECIDED: tuple[Decided, ...] = (
    Decided(
        id="w12-t5",
        summary="generated present-as-vacant duplicate mutation",
        reason=(
            "the documented behaviour (treat a located present key as vacant and "
            "insert a duplicate) is the same generated-body restructuring as "
            "w12-t4, which is encoded at the proof level by mutating "
            "`replacePresent`. The plan records no distinct exact-string or "
            "distinct model target for w12-t5, so there is no faithful second "
            "mutation; re-encoding the same defect would not add a control"
        ),
    ),
)

CONTROL_BY_ID = {control.id: control for control in CONTROLS}
DECIDED_BY_ID = {control.id: control for control in DECIDED}
KNOWN_IDS = frozenset(CONTROL_BY_ID) | frozenset(DECIDED_BY_ID)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_status_hash() -> str:
    """Hash ``git status --porcelain`` so a working-tree write is detectable."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise ControlError(
            f"git status --porcelain failed ({result.returncode}): {result.stderr.strip()}"
        )
    return hashlib.sha256(result.stdout.encode("utf-8")).hexdigest()


def first_error_locator(output: str) -> Optional[tuple[str, int, int]]:
    """Return the first ``file:line:col`` error in a ``lake build`` transcript."""
    match = ERROR_RE.search(output)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def enclosing_declaration(text: str, line: int) -> Optional[str]:
    """Name of the last top-level declaration at or before ``line``."""
    name: Optional[str] = None
    for index, content in enumerate(text.splitlines(), start=1):
        if index > line:
            break
        match = DECL_RE.match(content)
        if match is not None:
            name = match.group(1)
    return name


def error_declaration(dest: Path, file: str, line: int) -> Optional[str]:
    """Resolve the declaration enclosing ``file:line`` in the throwaway copy."""
    for base in (dest / "proof" / "corpus", dest / "proof"):
        candidate = base / file
        if candidate.is_file():
            return enclosing_declaration(candidate.read_text(), line)
    return None


def require_tool(name: str, which=shutil.which) -> Optional[str]:
    """Return an error message when ``name`` is absent from ``PATH``, else None."""
    if which(name) is None:
        return f"required tool missing: {name}"
    return None


# ---------------------------------------------------------------------------
# Throwaway tree construction
# ---------------------------------------------------------------------------


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=True)


def prepare_throwaway(control: Control, root: Path) -> Path:
    """Create a per-control build copy that cannot write through to the repo."""
    dest = root / control.id
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "proof").mkdir(parents=True)
    (dest / "proof" / "corpus").mkdir(parents=True)

    # proof package: real-copy the sources and the small build cache, symlink the
    # read-only dependency cache.
    for name in (
        "ApeironWord.lean",
        "Denotation.lean",
        "Refinement.lean",
        "lakefile.lean",
        "lake-manifest.json",
        "lean-toolchain",
    ):
        shutil.copy2(REPO_ROOT / "proof" / name, dest / "proof" / name)
    _copy_tree(REPO_ROOT / "proof" / "extract", dest / "proof" / "extract")
    shutil.copy2(REPO_ROOT / "proof" / "regen.sh", dest / "proof" / "regen.sh")
    (dest / "proof" / ".lake").mkdir()
    _copy_tree(REPO_ROOT / "proof" / ".lake" / "build", dest / "proof" / ".lake" / "build")
    _copy_tree(REPO_ROOT / "proof" / ".lake" / "config", dest / "proof" / ".lake" / "config")
    os.symlink(
        (REPO_ROOT / "proof" / ".lake" / "packages").resolve(),
        dest / "proof" / ".lake" / "packages",
    )

    # corpus package
    for entry in sorted((REPO_ROOT / "proof" / "corpus").glob("*.lean")):
        shutil.copy2(entry, dest / "proof" / "corpus" / entry.name)
    for name in ("lakefile.lean", "lake-manifest.json", "lean-toolchain"):
        shutil.copy2(
            REPO_ROOT / "proof" / "corpus" / name,
            dest / "proof" / "corpus" / name,
        )
    (dest / "proof" / "corpus" / ".lake").mkdir()
    _copy_tree(
        REPO_ROOT / "proof" / "corpus" / ".lake" / "build",
        dest / "proof" / "corpus" / ".lake" / "build",
    )
    _copy_tree(
        REPO_ROOT / "proof" / "corpus" / ".lake" / "config",
        dest / "proof" / "corpus" / ".lake" / "config",
    )
    os.symlink(
        (REPO_ROOT / "proof" / "corpus" / ".lake" / "packages").resolve(),
        dest / "proof" / "corpus" / ".lake" / "packages",
    )

    # `proof/lakefile.lean` requires the pinned Aeneas Lean backend through
    # `../target/aeneas/backends/lean`; expose the repository `target/` read-only.
    os.symlink((REPO_ROOT / "target").resolve(), dest / "target")

    return dest


# ---------------------------------------------------------------------------
# Safety verification
# ---------------------------------------------------------------------------


def _newer_than(paths: Iterable[Path], start: float) -> list[str]:
    hits: list[str] = []
    for base in paths:
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            for name in filenames:
                candidate = Path(dirpath) / name
                try:
                    mtime = candidate.lstat().st_mtime
                except OSError:
                    continue
                if mtime > start:
                    hits.append(str(candidate))
    return hits


@dataclasses.dataclass(frozen=True)
class RepoSnapshot:
    """The repository identity a control must not change."""

    generated: str
    source: str
    status: str

    @classmethod
    def capture(cls, source_file: str) -> "RepoSnapshot":
        return cls(
            generated=sha256_file(REPO_ROOT / GENERATED_FILE),
            source=sha256_file(REPO_ROOT / source_file),
            status=git_status_hash(),
        )


def assert_repo_untouched(before: RepoSnapshot, source_file: str, started: float) -> None:
    """Assert the generated source, mutated source, working tree and caches are unchanged."""
    now = RepoSnapshot.capture(source_file)
    if now.generated != before.generated:
        raise ControlError(
            f"repository {GENERATED_FILE} changed during the control: "
            f"{before.generated} -> {now.generated}"
        )
    if now.source != before.source:
        raise ControlError(
            f"repository {source_file} changed during the control: "
            f"{before.source} -> {now.source}"
        )
    if now.status != before.status:
        raise ControlError(
            "repository working tree changed during the control: "
            f"git status digest {before.status} -> {now.status}"
        )
    written = _newer_than((REPO_ROOT / p for p in PACKAGE_DIRS), started)
    if written:
        raise ControlError(
            "read-only package cache was written during the control: "
            + ", ".join(written[:5])
        )


def assert_regular_copy(dest: Path, source_file: str) -> None:
    for relative in (source_file, GENERATED_FILE):
        target = dest / relative
        if target.is_symlink():
            raise ControlError(f"throwaway source is a symlink, not a copy: {target}")
        if not target.is_file():
            raise ControlError(f"throwaway source missing: {target}")


# ---------------------------------------------------------------------------
# Build execution
# ---------------------------------------------------------------------------


def run_lake_build(corpus_dir: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run the whole-package default target in ``corpus_dir``."""
    return subprocess.run(
        ["lake", "build"],
        cwd=corpus_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def check_red(
    control: Control, result: subprocess.CompletedProcess, dest: Path
) -> tuple[str, int, int, Optional[str]]:
    """Assert the build failed at the control's expected declaration."""
    if result.returncode == 0:
        raise ControlError(
            f"mutation survived: lake build exited 0 (expected failure at {control.expect})"
        )
    observed = first_error_locator(result.stdout)
    if observed is None:
        raise ControlError(
            f"lake build failed without a located error; expected {control.expect}\n"
            + result.stdout[-2000:]
        )
    file, line, col = observed
    name = error_declaration(dest, file, line)
    if file != control.expect.module or name != control.expect.name:
        raise ControlError(
            f"first error at {file}:{line}:{col} resolves to declaration {name!r}, "
            f"expected {control.expect}"
        )
    return file, line, col, name


def check_green(control: Control, result: subprocess.CompletedProcess) -> None:
    if result.returncode != 0:
        raise ControlError(
            f"restored build failed (exit {result.returncode})\n" + result.stdout[-2000:]
        )


# ---------------------------------------------------------------------------
# Control execution
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ControlResult:
    control_id: str
    status: str
    expect: str
    recorded: str
    observed: Optional[str] = None
    observed_name: Optional[str] = None
    red_exit: Optional[int] = None
    green_exit: Optional[int] = None
    detail: str = ""
    seconds: float = 0.0
    throwaway: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)


def execute_control(control: Control, root: Path, timeout: int, keep: bool) -> ControlResult:
    started = time.time()
    result = ControlResult(
        control_id=control.id,
        status="fail",
        expect=str(control.expect),
        recorded=control.recorded,
    )
    before = RepoSnapshot.capture(control.source_file)
    dest = prepare_throwaway(control, root)
    result.throwaway = str(dest)
    assert_regular_copy(dest, control.source_file)
    source = dest / control.source_file

    try:
        pristine = source.read_text()
        mutated = control.mutation.apply(pristine)
        source.write_text(mutated)

        red = run_lake_build(dest / "proof" / "corpus", timeout)
        result.red_exit = red.returncode
        file, line, col, name = check_red(control, red, dest)
        result.observed = f"{file}:{line}:{col}"
        result.observed_name = name

        source.write_text(pristine)
        green = run_lake_build(dest / "proof" / "corpus", timeout)
        result.green_exit = green.returncode
        check_green(control, green)

        assert_repo_untouched(before, control.source_file, started)
        result.status = "pass"
        result.detail = "red then green at the expected declaration"
    except (ControlError, subprocess.TimeoutExpired) as exc:
        result.detail = str(exc)
    finally:
        result.seconds = round(time.time() - started, 2)
        if not keep:
            shutil.rmtree(dest, ignore_errors=True)
    return result


def run_controls(
    controls: Iterable[Control], root: Path, jobs: int, timeout: int, keep: bool
) -> list[ControlResult]:
    selected = list(controls)
    if jobs <= 1:
        return [execute_control(c, root, timeout, keep) for c in selected]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(execute_control, c, root, timeout, keep): c for c in selected
        }
        return [future.result() for future in concurrent.futures.as_completed(futures)]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _synthetic(control: Control, expect: Expect) -> Control:
    return dataclasses.replace(control, expect=expect)


def self_test() -> int:
    """Mutate the runner's own expected declaration and confirm it fires."""
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    sample_text = (
        "namespace Demo\n"
        "\n"
        "def alpha : Nat := 0\n"
        "\n"
        "theorem beta (x : Nat) : x = x := by\n"
        "  rfl\n"
        "\n"
        "private theorem gamma : True := by\n"
        "  trivial\n"
    )
    record(
        "enclosing_declaration_theorem",
        enclosing_declaration(sample_text, 6) == "beta",
        f"line 6 -> {enclosing_declaration(sample_text, 6)!r}",
    )
    record(
        "enclosing_declaration_private",
        enclosing_declaration(sample_text, 9) == "gamma",
        f"line 9 -> {enclosing_declaration(sample_text, 9)!r}",
    )
    record(
        "enclosing_declaration_def",
        enclosing_declaration(sample_text, 3) == "alpha",
        f"line 3 -> {enclosing_declaration(sample_text, 3)!r}",
    )

    # A throwaway copy with a synthetic module so `check_red` can resolve a name.
    probe = Path(tempfile.mkdtemp(prefix="storage-controls-self-", dir="/tmp/opencode"))
    (probe / "proof" / "corpus").mkdir(parents=True)
    (probe / "proof" / "corpus" / "Synthetic.lean").write_text(sample_text)
    base = CONTROLS[0]
    synthetic = dataclasses.replace(
        base, source_file="proof/corpus/Synthetic.lean",
        expect=Expect("Synthetic.lean", "beta"),
    )
    output = "error: Synthetic.lean:6:2: unsolved goals\nLean exited with code 1\n"

    # 1. the assertion accepts the true declaration
    try:
        check_red(synthetic, subprocess.CompletedProcess([], 1, output), probe)
        record("red_accepts_matching", True, "matching declaration accepted")
    except ControlError as exc:  # pragma: no cover - failure path
        record("red_accepts_matching", False, str(exc))

    # 2. mutate the runner's own expected name and confirm the assertion fires
    wrong_name = _synthetic(synthetic, Expect("Synthetic.lean", "alpha"))
    try:
        check_red(wrong_name, subprocess.CompletedProcess([], 1, output), probe)
        record("red_rejects_wrong_name", False, "assertion did not fire")
    except ControlError as exc:
        record("red_rejects_wrong_name", True, f"assertion fired: {exc}")

    # 3. mutate the expected module and confirm the assertion fires
    wrong_module = _synthetic(synthetic, Expect("Other.lean", "beta"))
    try:
        check_red(wrong_module, subprocess.CompletedProcess([], 1, output), probe)
        record("red_rejects_wrong_module", False, "assertion did not fire")
    except ControlError as exc:
        record("red_rejects_wrong_module", True, f"assertion fired: {exc}")

    # 4. a surviving mutation (exit 0) is rejected
    try:
        check_red(synthetic, subprocess.CompletedProcess([], 0, "Build completed\n"), probe)
        record("red_rejects_survivor", False, "assertion did not fire")
    except ControlError as exc:
        record("red_rejects_survivor", True, f"assertion fired: {exc}")

    # 5. an unlocated failure is rejected
    try:
        check_red(synthetic, subprocess.CompletedProcess([], 1, "error: build failed\n"), probe)
        record("red_rejects_unlocated", False, "assertion did not fire")
    except ControlError as exc:
        record("red_rejects_unlocated", True, f"assertion fired: {exc}")

    shutil.rmtree(probe, ignore_errors=True)

    # 6. the mutation engine replaces exactly and fails loudly when absent
    sample = "alpha\nbeta\n"
    try:
        assert Mutation(old="beta", new="gamma").apply(sample) == "alpha\ngamma\n"
        record("mutation_exact_replace", True, "exact replacement applied")
    except MutationError as exc:  # pragma: no cover - failure path
        record("mutation_exact_replace", False, str(exc))

    try:
        Mutation(old="absent", new="x").apply(sample)
        record("mutation_absent_raises", False, "missing target did not raise")
    except MutationError as exc:
        record("mutation_absent_raises", True, f"raised: {exc}")

    try:
        Mutation(anchor="alpha", old="beta", new="gamma").apply(sample)
        record("mutation_scoped_replace", True, "scoped replacement applied")
    except MutationError as exc:  # pragma: no cover - failure path
        record("mutation_scoped_replace", False, str(exc))

    try:
        Mutation(anchor="alpha", old="absent", new="x").apply(sample)
        record("mutation_scoped_absent_raises", False, "missing scoped target did not raise")
    except MutationError as exc:
        record("mutation_scoped_absent_raises", True, f"raised: {exc}")

    try:
        Mutation(anchor="nope", old="beta", new="x").apply(sample)
        record("mutation_missing_anchor_raises", False, "missing anchor did not raise")
    except MutationError as exc:
        record("mutation_missing_anchor_raises", True, f"raised: {exc}")

    # 7. locator parsing picks the first located error, not the summary lines
    parsed = first_error_locator(
        "error: Lean exited with code 1\nerror: A.lean:7:2: boom\n"
    )
    record(
        "first_error_locator",
        parsed == ("A.lean", 7, 2),
        f"parsed={parsed}",
    )

    # 8. every encoded mutation resolves against the real source and changes it
    for control in CONTROLS:
        source_path = REPO_ROOT / control.source_file
        if not source_path.is_file():
            record(
                f"mutation_resolves:{control.id}",
                False,
                f"missing source {control.source_file}",
            )
            continue
        try:
            pristine = source_path.read_text()
            mutated = control.mutation.apply(pristine)
            record(
                f"mutation_resolves:{control.id}",
                mutated != pristine,
                "target resolves uniquely and changes the source",
            )
        except (MutationError, OSError) as exc:
            record(f"mutation_resolves:{control.id}", False, str(exc))

    # 9. the decided control is rejected rather than silently dropped
    selected = _select(["w12-t5"])
    record(
        "decided_rejected_select",
        selected == [] and "w12-t5" in DECIDED_BY_ID,
        f"selected={selected}",
    )
    decided_cli = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--only", "w12-t5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    record(
        "decided_rejected_cli_exit_2",
        decided_cli.returncode == 2,
        f"returncode={decided_cli.returncode}",
    )
    record(
        "decided_rejected_cli_reason",
        "not encoded" in decided_cli.stdout and DECIDED_BY_ID["w12-t5"].reason[:40]
        in decided_cli.stdout,
        decided_cli.stdout.strip()[:200],
    )

    # 10. the missing-tool gate reports the absent executable and returns exit 3.
    message = require_tool("__apeiron_absent_tool__", which=lambda _name: None)
    record(
        "missing_tool_detected",
        message == "required tool missing: __apeiron_absent_tool__",
        f"message={message!r}",
    )
    try:
        require_tool("__apeiron_absent_tool__", which=lambda _name: None)
        present = require_tool("lake", which=lambda _name: "/usr/bin/lake")
        record("missing_tool_gate_present_ok", present is None, "present tool accepted")
    except ControlError as exc:  # pragma: no cover - failure path
        record("missing_tool_gate_present_ok", False, str(exc))

    # 11. end to end: an empty PATH makes the CLI reject `lake` with exit 3 and no
    #     success summary. The subprocess re-enters main without --self-test.
    empty_path = Path(tempfile.mkdtemp(prefix="storage-controls-empty-path-", dir="/tmp/opencode"))
    env = dict(os.environ)
    env["PATH"] = str(empty_path)
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--only", "w11-t5"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
    finally:
        shutil.rmtree(empty_path, ignore_errors=True)
    out = proc.stdout
    record(
        "missing_tool_cli_exit_3",
        proc.returncode == 3,
        f"returncode={proc.returncode}",
    )
    record(
        "missing_tool_cli_message",
        "required tool missing: lake" in out,
        out.strip()[:200],
    )
    record(
        "missing_tool_cli_no_success",
        '"summary"' not in out and '"pass"' not in out,
        out.strip()[:200],
    )

    for check in checks:
        print(json.dumps(check, sort_keys=True))
    failed = [c for c in checks if not c["ok"]]
    print(
        json.dumps(
            {"self_test": "pass" if not failed else "fail", "checks": len(checks)},
            sort_keys=True,
        )
    )
    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _select(only: list[str]) -> list[Control]:
    if not only:
        return list(CONTROLS)
    wanted: list[str] = []
    for item in only:
        wanted.extend(part for part in item.split(",") if part)
    unknown = [w for w in wanted if w not in KNOWN_IDS]
    if unknown:
        raise SystemExit(f"unknown control id(s): {', '.join(unknown)}")
    return [CONTROL_BY_ID[w] for w in wanted if w in CONTROL_BY_ID]


def cmd_list() -> int:
    for control in CONTROLS:
        print(f"{control.id}\tencoded\tmodule={control.expect.module}"
              f"\tname={control.expect.name}\trecorded={control.recorded}")
        print(f"    {control.summary}")
        if control.note:
            print(f"    note: {control.note}")
    for control in DECIDED:
        print(f"{control.id}\tdecided")
        print(f"    {control.summary}")
        print(f"    reason: {control.reason}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="storage-controls.py",
        description="Run the fixed-shard storage mutation controls in throwaway copies.",
    )
    parser.add_argument("--list", action="store_true", help="list controls and exit")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="ID",
        help="run only this control id (repeatable, comma-separated)",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="number of controls to run in parallel"
    )
    parser.add_argument(
        "--throwaway-root",
        type=Path,
        default=DEFAULT_THROWAWAY_ROOT,
        help=f"root for throwaway copies (default: {DEFAULT_THROWAWAY_ROOT})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT, help="per-build timeout in seconds"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove each throwaway copy after the control",
    )
    parser.add_argument("--self-test", action="store_true", help="verify the runner's own logic")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.list:
        return cmd_list()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    missing = require_tool("lake")
    if missing is not None:
        print(json.dumps({"error": missing}, sort_keys=True))
        return 3

    selected = _select(args.only)
    if not selected:
        requested = [w for item in args.only for w in item.split(",") if w]
        decided = [w for w in requested if w in DECIDED_BY_ID]
        if decided:
            for control_id in decided:
                print(
                    json.dumps(
                        {
                            "error": f"control {control_id} is not encoded",
                            "reason": DECIDED_BY_ID[control_id].reason,
                        },
                        sort_keys=True,
                    )
                )
        else:
            print(json.dumps({"error": "no controls selected"}, sort_keys=True))
        return 2

    args.throwaway_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = run_controls(
        selected, args.throwaway_root, args.jobs, args.timeout, keep=not args.clean
    )
    results.sort(key=lambda r: r.control_id)

    for result in results:
        print(result.to_json())

    failures = [r for r in results if r.status != "pass"]
    summary = {
        "total": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "seconds": round(time.time() - started, 2),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
