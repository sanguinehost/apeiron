#!/usr/bin/env python3
"""Fail-closed staged Slang source-refinement gate harness.

For one selected stage the harness checks the stage's artifacts and runs its
acceptance command; with ``--controls`` it also runs a baseline / mutation /
restored-copy control in a throwaway copy of the tree. The repository working
tree is never mutated.

Fail-closed contract
====================
Acceptance requires every one of the following; each failure is a named red
result with a nonzero exit status:

* the global source manifest resolves and every selected digest matches the
  working-tree bytes;
* every declared artifact resolves -- a missing proof module or source input is
  a blocked stage;
* every proof target declaration is **actual Lean code**: the module under test
  is built in the throwaway, then a generated probe imports the built module and
  finds the declaration in the elaborated environment. A declaration that exists
  only as commented text is absent from the environment and blocks the stage;
* the stage command starts and exits 0; a timeout or nonzero exit is blocked;
* with ``--controls`` the mutation turns the command red at the declared step
  and with the declared rejecting output, and the restored copy turns it green.
  A surviving mutation, a red run at the wrong step, or a red run with an
  unrelated result is a failed stage. A control marked ``awaiting-definition``
  records the plan's exact intended control and never passes ``--controls``; it
  is never replaced by a non-falsifying mutation.

Resolution and isolation
========================
A declared path resolves against exactly one root: a ``song/``-prefixed path
resolves only under ``--song-root`` (the explicit root wins and is the only
source root); every other path resolves only under the repository root. Before
any write the harness resolves the target and requires it inside the resolved
throwaway root, so a symlinked ancestor cannot redirect a mutation into the
working tree.

The harness establishes gate consistency only; it never claims correspondence,
refinement or proof. The eight stage ids bind to W22..W29 of
`the design notes/slang-source-refinement.md``.

Usage::

    python3 tools/slang/refine.py --list
    python3 tools/slang/refine.py --stage decode --controls
    python3 tools/slang/refine.py --stage init --stage lookup --song-root ../song
    python3 tools/slang/refine.py --self-test

Exit status: 0 when every selected stage passes; 1 when any stage is blocked or
fails; 2 on a usage error; 3 on a missing required tool.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
CONTROLS_PATH = HERE.parent / "controls.json"
DEFAULT_SONG_ROOT = REPO_ROOT.parent / "song"
DEFAULT_THROWAWAY_ROOT = Path("/tmp/apeiron/slang-refine")
DEFAULT_TIMEOUT = 1800

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_USAGE = 2
EXIT_TOOL = 3

HEX64 = re.compile(r"[0-9a-f]{64}")
STAGE_ID = re.compile(r"[A-Za-z0-9._-]+")
PROBE_NAME = "__ApeironDeclProbe.lean"
SELFTEST_TOOLCHAIN = "leanprover/lean4:v4.31.0"
SOURCE_ARTIFACT = "source"
PROOF_ARTIFACT = "proof"
ACTIVE = "active"
AWAITING = "awaiting-definition"

class GateError(Exception):
    """A stage could not reach its acceptance condition."""

class ArtifactError(GateError):
    """A declared artifact is missing, stale or not actual Lean code."""

class CommandTimeout(GateError):
    """A stage command exceeded its declared timeout."""

class CommandFailure(GateError):
    """A stage command could not start or did not exit 0 when required."""

class MutationError(GateError):
    """The declared mutation target was absent or ambiguous."""

class ControlsError(GateError):
    """``controls.json`` is missing or malformed."""

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Artifact:
    kind: str
    root: str | None = None
    module: str | None = None
    declaration: str | None = None
    declaration_prefix: str | None = None
    path: str | None = None
    sha256: str | None = None
    note: str = ""

@dataclasses.dataclass(frozen=True)
class Mutation:
    path: str
    old: str
    new: str
    anchor: str | None = None
    expect: str = ""
    expected_step: int = 0
    expected_output: str = ""

    def apply(self, text: str) -> str:
        """Apply the mutation; a missing or ambiguous target is a hard error."""
        if self.anchor is None:
            count = text.count(self.old)
            if count != 1:
                raise MutationError(
                    f"mutation target occurs {count} times, expected exactly 1: "
                    f"{self.old!r}"
                )
            return text.replace(self.old, self.new, 1)
        anchor_count = text.count(self.anchor)
        if anchor_count != 1:
            raise MutationError(
                f"mutation anchor occurs {anchor_count} times, expected exactly 1: "
                f"{self.anchor!r}"
            )
        start = text.index(self.anchor)
        at = text.find(self.old, start)
        if at < 0:
            raise MutationError(
                f"scoped mutation target absent after anchor {self.anchor!r}: "
                f"{self.old!r}"
            )
        return text[:at] + self.new + text[at + len(self.old) :]

@dataclasses.dataclass(frozen=True)
class Command:
    steps: tuple[tuple[str, ...], ...]
    cwd: str
    timeout: int

@dataclasses.dataclass(frozen=True)
class Stage:
    id: str
    order: int
    title: str
    summary: str
    source_entry: str
    rejecting_control: str
    artifacts: tuple[Artifact, ...]
    command: Command
    mutation: Mutation | None
    copy: tuple[str, ...]
    link: tuple[str, ...]
    control_status: str = ACTIVE
    awaiting_reason: str = ""
    note: str = ""

@dataclasses.dataclass(frozen=True)
class Controls:
    roots: dict[str, str]
    manifest: tuple[Artifact, ...]
    stages: tuple[Stage, ...]

@dataclasses.dataclass(frozen=True)
class RunResult:
    step: int
    argv: tuple[str, ...]
    returncode: int
    output: str

@dataclasses.dataclass
class StageResult:
    stage: str
    status: str
    reason: str
    baseline: str = "not-run"
    red_step: int | None = None
    red_exit: int | None = None
    green_exit: int | None = None
    throwaway: str | None = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text[-limit:]

def _inside(candidate: Path, root: Path) -> bool:
    """True when ``candidate`` resolves to ``root`` or a descendant of it."""
    try:
        resolved = candidate.resolve()
        resolved_root = root.resolve()
    except OSError:
        return False
    try:
        return os.path.commonpath([str(resolved), str(resolved_root)]) == str(
            resolved_root
        )
    except ValueError:
        return False

def _split_root(path: str) -> tuple[str, str]:
    """Classify a declared path to exactly one source root.

    ``song`` and ``song/<rest>`` resolve only under the explicit Song root;
    every other path resolves only under the repository root. There is no
    fallback to an ordinary sibling checkout.
    """
    if path == "song":
        return "song", "."
    if path.startswith("song/"):
        return "song", path[len("song/") :]
    return "repo", path

def resolve_many(path: str, repo_root: Path, song_root: Path) -> list[Path]:
    """Resolve a declared path under exactly one root; globs expand in place."""
    kind, relative = _split_root(path)
    base = song_root if kind == "song" else repo_root
    matches: list[Path] = []
    if any(ch in relative for ch in "*?["):
        try:
            matches.extend(sorted(base.glob(relative)))
        except (OSError, ValueError):
            return []
    else:
        candidate = base / relative
        if candidate.exists():
            matches.append(candidate)
    seen: set[Path] = set()
    unique: list[Path] = []
    for match in matches:
        try:
            resolved = match.resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(match)
    return unique

def resolve_one(path: str, repo_root: Path, song_root: Path) -> Path | None:
    matches = resolve_many(path, repo_root, song_root)
    return matches[0] if matches else None

# ---------------------------------------------------------------------------
# controls.json parsing
# ---------------------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlsError(message)

def _require_str(value: object, where: str) -> str:
    _require(
        isinstance(value, str) and value.strip() != "",
        f"{where} must be a nonempty string",
    )
    return value  # type: ignore[return-value]

def _parse_artifact(raw: object, where: str) -> Artifact:
    _require(isinstance(raw, dict), f"{where} must be an object")
    kind = _require_str(raw.get("kind"), f"{where}.kind")
    if kind == PROOF_ARTIFACT:
        root = _require_str(raw.get("root"), f"{where}.root")
        module = _require_str(raw.get("module"), f"{where}.module")
        declaration = raw.get("declaration")
        prefix = raw.get("declaration_prefix")
        _require(
            declaration is not None or prefix is not None,
            f"{where} must declare `declaration` or `declaration_prefix`",
        )
        if declaration is not None:
            _require_str(declaration, f"{where}.declaration")
        if prefix is not None:
            _require_str(prefix, f"{where}.declaration_prefix")
        return Artifact(
            kind=kind,
            root=root,
            module=module,
            declaration=declaration,
            declaration_prefix=prefix,
            note=str(raw.get("note", "")),
        )
    if kind == SOURCE_ARTIFACT:
        path = _require_str(raw.get("path"), f"{where}.path")
        digest = _require_str(raw.get("sha256"), f"{where}.sha256")
        _require(
            HEX64.fullmatch(digest) is not None,
            f"{where}.sha256 must be a 64-hex sha256 digest",
        )
        return Artifact(
            kind=kind, path=path, sha256=digest, note=str(raw.get("note", ""))
        )
    raise ControlsError(f"{where}.kind must be 'proof' or 'source', got {kind!r}")

def _parse_mutation(raw: object, where: str) -> Mutation:
    _require(isinstance(raw, dict), f"{where} must be an object")
    path = _require_str(raw.get("path"), f"{where}.path")
    old = _require_str(raw.get("old"), f"{where}.old")
    new = raw.get("new")
    _require(isinstance(new, str), f"{where}.new must be a string")
    _require(old != new, f"{where}.new must differ from {where}.old")
    anchor = raw.get("anchor")
    if anchor is not None:
        _require_str(anchor, f"{where}.anchor")
    expected_step = raw.get("expected_step")
    _require(
        isinstance(expected_step, int) and expected_step >= 0,
        f"{where}.expected_step must be a nonnegative integer",
    )
    expected_output = _require_str(raw.get("expected_output"), f"{where}.expected_output")
    return Mutation(
        path=path,
        old=old,
        new=new,
        anchor=anchor,
        expect=str(raw.get("expect", "")),
        expected_step=expected_step,
        expected_output=expected_output,
    )

def _parse_command(raw: object, where: str, default_timeout: int) -> Command:
    _require(isinstance(raw, dict), f"{where} must be an object")
    steps_raw = raw.get("steps")
    _require(
        isinstance(steps_raw, list) and steps_raw,
        f"{where}.steps must be a nonempty list of argv lists",
    )
    steps: list[tuple[str, ...]] = []
    for index, step in enumerate(steps_raw):
        _require(
            isinstance(step, list) and step,
            f"{where}.steps[{index}] must be a nonempty argv list",
        )
        argv = tuple(_require_str(token, f"{where}.steps[{index}]") for token in step)
        steps.append(argv)
    cwd = _require_str(raw.get("cwd"), f"{where}.cwd")
    timeout = raw.get("timeout", default_timeout)
    _require(
        isinstance(timeout, int) and timeout > 0,
        f"{where}.timeout must be a positive integer",
    )
    return Command(steps=tuple(steps), cwd=cwd, timeout=timeout)

def parse_stage(
    raw: object,
    where: str,
    *,
    roots: dict[str, str],
    default_timeout: int,
    default_copy: tuple[str, ...],
    default_link: tuple[str, ...],
) -> Stage:
    _require(isinstance(raw, dict), f"{where} must be an object")
    stage_id = _require_str(raw.get("id"), f"{where}.id")
    _require(
        STAGE_ID.fullmatch(stage_id) is not None and ".." not in stage_id,
        f"{where}.id {stage_id!r} is not a safe stage id",
    )
    order = raw.get("order")
    _require(isinstance(order, int) and order > 0, f"{where}.order must be positive")
    artifacts_raw = raw.get("artifacts")
    _require(
        isinstance(artifacts_raw, list) and artifacts_raw,
        f"{where}.artifacts must be a nonempty list",
    )
    artifacts = tuple(
        _parse_artifact(item, f"{where}.artifacts[{index}]")
        for index, item in enumerate(artifacts_raw)
    )
    for artifact in artifacts:
        if artifact.kind == PROOF_ARTIFACT:
            _require(
                artifact.root in roots,
                f"{where}: proof root {artifact.root!r} is not declared in `roots`",
            )
    command = _parse_command(raw.get("command"), f"{where}.command", default_timeout)
    control_status = str(raw.get("control_status", ACTIVE))
    _require(
        control_status in (ACTIVE, AWAITING),
        f"{where}.control_status must be {ACTIVE!r} or {AWAITING!r}",
    )
    mutation_raw = raw.get("mutation")
    if control_status == ACTIVE:
        _require(
            mutation_raw is not None,
            f"{where}: an active control requires a `mutation`",
        )
        mutation = _parse_mutation(mutation_raw, f"{where}.mutation")
        _require(
            mutation.expected_step < len(command.steps),
            f"{where}.mutation.expected_step {mutation.expected_step} is outside the "
            f"{len(command.steps)} declared command step(s)",
        )
        awaiting_reason = ""
    else:
        _require(
            mutation_raw is None,
            f"{where}: an awaiting-definition control must not declare a `mutation`",
        )
        mutation = None
        awaiting_reason = _require_str(
            raw.get("awaiting_reason"), f"{where}.awaiting_reason"
        )
    copy_raw = raw.get("copy", list(default_copy))
    link_raw = raw.get("link", list(default_link))
    _require(isinstance(copy_raw, list), f"{where}.copy must be a list")
    _require(isinstance(link_raw, list), f"{where}.link must be a list")
    return Stage(
        id=stage_id,
        order=order,
        title=_require_str(raw.get("title"), f"{where}.title"),
        summary=str(raw.get("summary", "")),
        source_entry=str(raw.get("source_entry", "")),
        rejecting_control=str(raw.get("rejecting_control", "")),
        artifacts=artifacts,
        command=command,
        mutation=mutation,
        copy=tuple(_require_str(entry, f"{where}.copy") for entry in copy_raw),
        link=tuple(_require_str(entry, f"{where}.link") for entry in link_raw),
        control_status=control_status,
        awaiting_reason=awaiting_reason,
        note=str(raw.get("note", "")),
    )

def parse_controls(data: object) -> Controls:
    _require(isinstance(data, dict), "controls.json must be an object")
    _require(data.get("schema") == 1, "controls.json schema must be 1")
    roots_raw = data.get("roots")
    _require(isinstance(roots_raw, dict) and roots_raw, "controls.json `roots` missing")
    roots: dict[str, str] = {}
    for name, value in roots_raw.items():
        roots[_require_str(name, "roots key")] = _require_str(value, f"roots.{name}")
    manifest_raw = data.get("manifest")
    _require(
        isinstance(manifest_raw, list) and manifest_raw,
        "controls.json `manifest` must be a nonempty list",
    )
    manifest = tuple(
        _parse_artifact(item, f"manifest[{index}]")
        for index, item in enumerate(manifest_raw)
    )
    for entry in manifest:
        _require(
            entry.kind == SOURCE_ARTIFACT,
            "controls.json `manifest` entries must be source artifacts",
        )
    defaults = data.get("defaults", {})
    _require(isinstance(defaults, dict), "controls.json `defaults` must be an object")
    default_timeout = defaults.get("timeout", DEFAULT_TIMEOUT)
    _require(
        isinstance(default_timeout, int) and default_timeout > 0,
        "defaults.timeout must be a positive integer",
    )
    default_copy = tuple(defaults.get("copy", []))
    default_link = tuple(defaults.get("link", []))
    stages_raw = data.get("stages")
    _require(
        isinstance(stages_raw, list) and stages_raw,
        "controls.json `stages` must be a nonempty list",
    )
    stages = [
        parse_stage(
            item,
            f"stages[{index}]",
            roots=roots,
            default_timeout=default_timeout,
            default_copy=default_copy,
            default_link=default_link,
        )
        for index, item in enumerate(stages_raw)
    ]
    ids = [stage.id for stage in stages]
    _require(len(ids) == len(set(ids)), f"duplicate stage id(s): {ids}")
    return Controls(roots=roots, manifest=manifest, stages=tuple(stages))

def load_controls(path: Path = CONTROLS_PATH) -> Controls:
    if not path.is_file():
        raise ControlsError(f"controls file missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ControlsError(f"controls file is not valid JSON: {exc}") from exc
    return parse_controls(data)

# ---------------------------------------------------------------------------
# Artifact checks
# ---------------------------------------------------------------------------

def check_source(path: str, digest: str, repo_root: Path, song_root: Path) -> None:
    resolved = resolve_one(path, repo_root, song_root)
    if resolved is None:
        raise ArtifactError(f"missing source input: {path}")
    actual = sha256_file(resolved)
    if actual != digest:
        raise ArtifactError(
            f"stale source input: {path} digest {actual} != recorded {digest}"
        )

def check_manifest(manifest: tuple[Artifact, ...], repo_root: Path, song_root: Path) -> None:
    for entry in manifest:
        check_source(entry.path, entry.sha256, repo_root, song_root)  # type: ignore[arg-type]

def check_artifact(
    artifact: Artifact, roots: dict[str, str], repo_root: Path, song_root: Path
) -> None:
    if artifact.kind == PROOF_ARTIFACT:
        base = repo_root / roots[artifact.root]  # type: ignore[index]
        module = base / artifact.module  # type: ignore[arg-type]
        if not module.is_file():
            raise ArtifactError(
                f"missing proof target: {artifact.root}:{artifact.module}"
            )
        return
    if artifact.kind == SOURCE_ARTIFACT:
        check_source(artifact.path, artifact.sha256, repo_root, song_root)  # type: ignore[arg-type]
        return
    raise ArtifactError(f"unknown artifact kind: {artifact.kind!r}")

def check_artifacts(
    stage: Stage, roots: dict[str, str], repo_root: Path, song_root: Path
) -> None:
    for artifact in stage.artifacts:
        check_artifact(artifact, roots, repo_root, song_root)

# ---------------------------------------------------------------------------
# Throwaway copies and command execution
# ---------------------------------------------------------------------------

def _guard_write(target: Path, resolved_root: Path) -> None:
    if not _inside(target, resolved_root):
        raise GateError(
            f"refusing to write outside the resolved throwaway root: {target}"
        )

def _replace(target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

def build_throwaway(
    stage: Stage, dest: Path, repo_root: Path, song_root: Path
) -> None:
    """Copy declared inputs into ``dest``; symlink the declared read-only trees.

    Every target is resolved and required to stay inside the resolved throwaway
    root before it is written, so a symlinked ancestor cannot redirect a write
    into the repository.
    """
    dest.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest.resolve()
    for entry in stage.copy:
        matches = resolve_many(entry, repo_root, song_root)
        if not matches:
            raise GateError(f"throwaway copy input missing: {entry}")
        globbed = any(ch in entry for ch in "*?[")
        parent = Path(entry).parent if globbed else None
        for match in matches:
            target = (dest / parent / match.name) if globbed else (dest / entry)
            _guard_write(target, resolved_dest)
            target.parent.mkdir(parents=True, exist_ok=True)
            _replace(target)
            if match.is_dir():
                shutil.copytree(match, target)
            else:
                shutil.copy2(match, target)
    for entry in stage.link:
        matches = resolve_many(entry, repo_root, song_root)
        if not matches:
            continue
        target = dest / entry
        _guard_write(target, resolved_dest)
        target.parent.mkdir(parents=True, exist_ok=True)
        _replace(target)
        os.symlink(matches[0].resolve(), target)

    if stage.mutation is None:
        return
    # The mutation target must be a real copy inside the throwaway, even when the
    # stage's copy manifest does not list it; a symlink or an escaping ancestor
    # would write through to the repository.
    mutation_target = dest / stage.mutation.path
    _guard_write(mutation_target, resolved_dest)
    if not mutation_target.exists() or mutation_target.is_symlink():
        mutation_source = resolve_one(stage.mutation.path, repo_root, song_root)
        if mutation_source is None:
            raise GateError(f"mutation source missing: {stage.mutation.path}")
        mutation_target.parent.mkdir(parents=True, exist_ok=True)
        if mutation_target.is_symlink():
            mutation_target.unlink()
        shutil.copy2(mutation_source, mutation_target)

def _expand(
    argv: tuple[str, ...], dest: Path, repo_root: Path, song_root: Path
) -> list[str]:
    return [
        token.replace("{song_root}", str(song_root))
        .replace("{repo}", str(repo_root))
        .replace("{root}", str(dest))
        for token in argv
    ]

def _run_once(
    argv: list[str], cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    """Run one argv, raising a named error on timeout or an unstartable tool."""
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandTimeout(
            f"timeout after {timeout}s running {' '.join(argv)}"
        ) from exc
    except OSError as exc:
        raise CommandFailure(
            f"subprocess failed to start ({' '.join(argv)}): {exc}"
        ) from exc

def run_steps(
    stage: Stage,
    dest: Path,
    repo_root: Path,
    song_root: Path,
    timeout_override: int | None = None,
) -> RunResult:
    """Run the stage steps in order; return the first nonzero result, else the last.

    A timeout or an unstartable executable raises a named :class:`GateError`.
    """
    timeout = timeout_override or stage.command.timeout
    cwd = dest / stage.command.cwd
    result: RunResult | None = None
    for index, argv in enumerate(stage.command.steps):
        expanded = _expand(argv, dest, repo_root, song_root)
        completed = _run_once(expanded, cwd, timeout)
        result = RunResult(
            step=index,
            argv=tuple(expanded),
            returncode=completed.returncode,
            output=completed.stdout or "",
        )
        if completed.returncode != 0:
            return result
    assert result is not None
    return result

# ---------------------------------------------------------------------------
# Proof declaration freshness
# ---------------------------------------------------------------------------

def _lean_string(value: str) -> str:
    return json.dumps(value)

def _probe_text(import_name: str, declaration: str, prefix: str) -> str:
    return (
        f"import {import_name}\n"
        "import Lean.Elab.Command\n"
        "open Lean Elab Command\n"
        "run_cmd do\n"
        "  let env ← getEnv\n"
        f"  let exact := {_lean_string(declaration)}\n"
        f"  let pfx := {_lean_string(prefix)}\n"
        "  let found := env.constants.toList.any (fun (n, _) =>\n"
        "    let s := n.toString\n"
        "    (exact != \"\" && s == exact) ||\n"
        "    (pfx != \"\" && (s == pfx || s.startsWith (pfx ++ \".\"))))\n"
        "  if !found then throwError \"missing proof target declaration\"\n"
        "  logInfo \"declaration-present\"\n"
    )

def check_proof_declarations(
    stage: Stage, roots: dict[str, str], dest: Path, timeout: int
) -> None:
    """Build each module under test, then require its declaration to be loaded.

    The declaration must be present in the elaborated environment of the built
    module, so a declaration that exists only as commented text is rejected.
    """
    resolved_dest = dest.resolve()
    for artifact in stage.artifacts:
        if artifact.kind != PROOF_ARTIFACT:
            continue
        package_dir = dest / roots[artifact.root]  # type: ignore[index]
        if not _inside(package_dir, resolved_dest):
            raise ArtifactError(
                f"proof package root escapes the throwaway: {artifact.root}"
            )
        if not package_dir.is_dir():
            raise ArtifactError(
                f"missing proof package in throwaway: {artifact.root}"
            )
        module_rel = artifact.module  # type: ignore[assignment]
        module_path = package_dir / module_rel
        if not _inside(module_path, resolved_dest) or not module_path.is_file():
            raise ArtifactError(
                f"missing proof target: {artifact.root}:{module_rel}"
            )
        import_name = (
            module_rel[:-5].replace("/", ".")
            if module_rel.endswith(".lean")
            else module_rel
        )
        exact = artifact.declaration or ""
        prefix = artifact.declaration_prefix or ""
        probe = package_dir / PROBE_NAME
        _guard_write(probe, resolved_dest)
        probe.write_text(_probe_text(import_name, exact, prefix), encoding="utf-8")
        try:
            build = _run_once(["lake", "build", import_name], package_dir, timeout)
            if build.returncode != 0:
                raise ArtifactError(
                    f"proof dependency build failed: {artifact.root}:{module_rel}: "
                    f"{_tail(build.stdout)}"
                )
            loaded = _run_once(
                ["lake", "env", "lean", PROBE_NAME], package_dir, timeout
            )
            if loaded.returncode != 0:
                raise ArtifactError(
                    f"proof target declaration not loaded from real code: "
                    f"{artifact.root}:{module_rel}::{exact or prefix}: "
                    f"{_tail(loaded.stdout)}"
                )
        finally:
            probe.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------

def check_stage(
    stage: Stage,
    roots: dict[str, str],
    manifest: tuple[Artifact, ...],
    repo_root: Path,
    song_root: Path,
    *,
    controls: bool,
    throwaway_root: Path,
    keep: bool = False,
    timeout_override: int | None = None,
) -> StageResult:
    result = StageResult(stage=stage.id, status="blocked", reason="not-run")
    try:
        check_manifest(manifest, repo_root, song_root)
        check_artifacts(stage, roots, repo_root, song_root)
    except ArtifactError as exc:
        result.reason = str(exc)
        return result

    dest = throwaway_root / stage.id
    result.throwaway = str(dest)
    try:
        if dest.exists():
            shutil.rmtree(dest)
        build_throwaway(stage, dest, repo_root, song_root)
    except (GateError, OSError) as exc:
        result.reason = f"throwaway copy failed: {exc}"
        return result

    try:
        check_proof_declarations(
            stage, roots, dest, timeout_override or stage.command.timeout
        )
    except (GateError, OSError) as exc:
        result.reason = str(exc)
        _cleanup(dest, keep)
        return result

    if controls and stage.control_status != ACTIVE:
        result.reason = f"control {stage.control_status}: {stage.awaiting_reason}"
        _cleanup(dest, keep)
        return result
    if stage.mutation is None:
        result.reason = "no active control declared"
        _cleanup(dest, keep)
        return result

    mutation = stage.mutation
    mutated_path = dest / mutation.path
    resolved_dest = dest.resolve()
    try:
        _guard_write(mutated_path, resolved_dest)
        pristine = mutated_path.read_bytes()
    except (GateError, OSError) as exc:
        result.reason = f"mutation target unusable in throwaway: {exc}"
        _cleanup(dest, keep)
        return result

    try:
        baseline = run_steps(stage, dest, repo_root, song_root, timeout_override)
    except GateError as exc:
        result.reason = str(exc)
        _cleanup(dest, keep)
        return result
    if baseline.returncode != 0:
        result.reason = (
            f"nonzero subprocess (exit {baseline.returncode}) in baseline: "
            f"{_tail(baseline.output)}"
        )
        _cleanup(dest, keep)
        return result
    result.baseline = "green"
    if not controls:
        result.status = "pass"
        result.reason = "baseline green; controls not requested"
        _cleanup(dest, keep)
        return result

    try:
        mutated = mutation.apply(pristine.decode("utf-8"))
    except (MutationError, UnicodeDecodeError) as exc:
        result.status = "fail"
        result.reason = f"mutation rejected: {exc}"
        _cleanup(dest, keep)
        return result
    try:
        _guard_write(mutated_path, resolved_dest)
        mutated_path.write_text(mutated, encoding="utf-8")
    except (GateError, OSError) as exc:
        result.status = "fail"
        result.reason = f"mutation write failed: {exc}"
        _cleanup(dest, keep)
        return result

    try:
        red = run_steps(stage, dest, repo_root, song_root, timeout_override)
    except GateError as exc:
        result.status = "fail"
        result.reason = f"mutation run failed: {exc}"
        _cleanup(dest, keep)
        return result
    result.red_step = red.step
    result.red_exit = red.returncode
    if red.returncode == 0:
        result.status = "fail"
        result.reason = (
            "mutation survived: command exited 0 (expected red at "
            f"{mutation.expect or 'the declared mutation'})"
        )
        _cleanup(dest, keep)
        return result
    if red.step != mutation.expected_step:
        result.status = "fail"
        result.reason = (
            f"control target not reached: mutation failed at step {red.step}, "
            f"expected step {mutation.expected_step} ({mutation.expect or 'declared'})"
        )
        _cleanup(dest, keep)
        return result
    if mutation.expected_output not in red.output:
        result.status = "fail"
        result.reason = (
            "mutation rejected without the expected rejecting result "
            f"{mutation.expected_output!r}: {_tail(red.output)}"
        )
        _cleanup(dest, keep)
        return result

    try:
        _guard_write(mutated_path, resolved_dest)
        mutated_path.write_bytes(pristine)
    except (GateError, OSError) as exc:
        result.status = "fail"
        result.reason = f"restore failed: {exc}"
        _cleanup(dest, keep)
        return result
    try:
        green = run_steps(stage, dest, repo_root, song_root, timeout_override)
    except GateError as exc:
        result.status = "fail"
        result.reason = f"restored run failed: {exc}"
        _cleanup(dest, keep)
        return result
    result.green_exit = green.returncode
    if green.returncode != 0:
        result.status = "fail"
        result.reason = (
            f"restored copy failed (exit {green.returncode}): {_tail(green.output)}"
        )
        _cleanup(dest, keep)
        return result

    result.status = "pass"
    result.reason = (
        f"baseline green, mutation red at step {red.step} "
        f"(exit {red.returncode}) with the expected result, restored green"
    )
    _cleanup(dest, keep)
    return result

def _cleanup(dest: Path, keep: bool) -> None:
    if not keep:
        shutil.rmtree(dest, ignore_errors=True)

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _synthetic_repo(root: Path) -> Path:
    (root / "Slang").mkdir(parents=True, exist_ok=True)
    (root / "song").mkdir(parents=True, exist_ok=True)
    (root / "Slang" / "Synthetic.lean").write_text(
        "namespace Slang.Synthetic\n\ndef ok : Nat := 1\n\nend Slang.Synthetic\n",
        encoding="utf-8",
    )
    (root / "Slang" / "Comment.lean").write_text(
        "namespace Slang.Comment\n\n/-\ndef gone : Nat := 1\n-/\n\nend Slang.Comment\n",
        encoding="utf-8",
    )
    (root / "Slang" / "Broken.lean").write_text(
        "namespace Slang.Broken\n\ndef bad : Nat := \"not a nat\"\n\nend Slang.Broken\n",
        encoding="utf-8",
    )
    (root / "lakefile.lean").write_text(
        "import Lake\nopen Lake DSL\n\npackage \"synthetic\"\n\n"
        "lean_lib «Synthetic» where\n  globs := #[.submodules `Slang]\n",
        encoding="utf-8",
    )
    (root / "lean-toolchain").write_text(SELFTEST_TOOLCHAIN + "\n", encoding="utf-8")
    (root / "song" / "synthetic.slang").write_text(
        "body { return 1; }\n", encoding="utf-8"
    )
    (root / "check.py").write_text(
        "import pathlib, sys\n"
        "text = pathlib.Path('song/synthetic.slang').read_text()\n"
        "if 'return 2;' in text:\n"
        "    print('synthetic body change rejected')\n"
        "    sys.exit(2)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    (root / "crash.py").write_text(
        "import pathlib, sys\n"
        "text = pathlib.Path('song/synthetic.slang').read_text()\n"
        "if 'boom' in text:\n"
        "    print('unrelated crash')\n"
        "    sys.exit(3)\n"
        "sys.exit(2 if 'return 2;' in text else 0)\n",
        encoding="utf-8",
    )
    return root

SYNTHETIC_COPY = [
    "song/synthetic.slang",
    "Slang/Synthetic.lean",
    "Slang/Comment.lean",
    "Slang/Broken.lean",
    "lakefile.lean",
    "lean-toolchain",
    "check.py",
    "crash.py",
]

def _synthetic_stage_dict(
    root: Path,
    *,
    module: str = "Slang/Synthetic.lean",
    prefix: str = "Slang.Synthetic",
    declaration: str | None = None,
    digest: str | None = None,
    steps: list[list[str]] | None = None,
    timeout: int = 60,
    old: str = "return 1;",
    new: str = "return 2;",
    expected_step: int = 0,
    expected_output: str = "synthetic body change rejected",
    control_status: str = ACTIVE,
    anchor: str | None = None,
    copy: list[str] | None = None,
    link: list[str] | None = None,
) -> dict:
    if digest is None:
        digest = sha256_file(root / "song" / "synthetic.slang")
    proof: dict = {"kind": "proof", "root": "bridge", "module": module}
    if declaration is not None:
        proof["declaration"] = declaration
    else:
        proof["declaration_prefix"] = prefix
    raw: dict = {
        "id": "synthetic",
        "order": 1,
        "title": "synthetic stage",
        "summary": "self-test synthetic stage",
        "source_entry": "SE0",
        "rejecting_control": "SC0",
        "artifacts": [
            proof,
            {"kind": "source", "path": "song/synthetic.slang", "sha256": digest},
        ],
        "command": {
            "cwd": ".",
            "timeout": timeout,
            "steps": steps or [["python3", "check.py"]],
        },
        "copy": list(copy if copy is not None else SYNTHETIC_COPY),
        "link": list(link if link is not None else []),
        "control_status": control_status,
    }
    if control_status == ACTIVE:
        mutation: dict = {
            "path": "song/synthetic.slang",
            "old": old,
            "new": new,
            "expect": "synthetic body change rejected",
            "expected_step": expected_step,
            "expected_output": expected_output,
        }
        if anchor is not None:
            mutation["anchor"] = anchor
        raw["mutation"] = mutation
    else:
        raw["awaiting_reason"] = "synthetic control awaiting its real definition"
    return raw

def self_test() -> int:
    """Prove the rejection classes fire and the positive accepts."""
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    def expect(
        name: str, result: StageResult, status: str, *needles: str, extra: bool = True
    ) -> None:
        ok = result.status == status and all(n in result.reason for n in needles) and extra
        record(name, ok, f"status={result.status} reason={result.reason}")

    workdir = Path(tempfile.mkdtemp(prefix="slang-refine-self-", dir="/tmp/apeiron"))
    try:
        repo = _synthetic_repo(workdir / "repo")
        roots = {"bridge": "."}
        default_manifest = (
            Artifact(
                kind=SOURCE_ARTIFACT,
                path="song/synthetic.slang",
                sha256=sha256_file(repo / "song" / "synthetic.slang"),
            ),
        )
        song_root = repo / "song"

        def stage_from(raw: dict) -> Stage:
            return parse_stage(
                raw,
                "synthetic",
                roots=roots,
                default_timeout=60,
                default_copy=(),
                default_link=(),
            )

        def run(
            raw: dict,
            *,
            controls: bool = True,
            manifest: tuple[Artifact, ...] | None = None,
        ) -> StageResult:
            return check_stage(
                stage_from(raw),
                roots,
                manifest if manifest is not None else default_manifest,
                repo,
                song_root,
                controls=controls,
                throwaway_root=workdir / "throwaway",
            )

        # 0. the positive control: fresh artifact, green baseline, mutation red at the
        #    declared step with the declared result, restored green.
        positive = run(_synthetic_stage_dict(repo), controls=True)
        record(
            "positive_present_and_fresh",
            positive.status == "pass"
            and positive.red_step == 0
            and positive.red_exit not in (None, 0)
            and positive.green_exit == 0,
            f"status={positive.status} reason={positive.reason}",
        )

        # 1. every simple rejection class, including hole 1 (a block-commented
        #    declaration is text, not Lean code) and dependency-building.
        cases = [
            ("reject_missing_artifact", {"module": "Slang/Absent.lean"}, "blocked", False, ("missing proof target",)),
            ("reject_missing_declaration", {"prefix": "Slang.Other"}, "blocked", False, ("not loaded from real code",)),
            ("reject_commented_declaration", {"module": "Slang/Comment.lean", "prefix": "Slang.Comment"}, "blocked", False, ("not loaded from real code",)),
            ("reject_stale_digest", {"digest": "0" * 64}, "blocked", False, ("stale source input",)),
            ("reject_timeout", {"steps": [["python3", "-c", "import time; time.sleep(30)"]], "timeout": 1}, "blocked", False, ("timeout",)),
            ("reject_nonzero_subprocess", {"steps": [["python3", "-c", "import sys; sys.exit(7)"]]}, "blocked", False, ("nonzero subprocess", "exit 7")),
            ("reject_unstartable_subprocess", {"steps": [["__apeiron_absent_tool__"]]}, "blocked", False, ("failed to start",)),
            ("reject_dependency_build_failure", {"module": "Slang/Broken.lean", "prefix": "Slang.Broken"}, "blocked", False, ("proof dependency build failed",)),
            ("reject_surviving_mutation", {"old": "body {", "new": "body  {"}, "fail", True, ("mutation survived",)),
            ("reject_awaiting_control", {"control_status": AWAITING}, "blocked", True, (AWAITING,)),
        ]
        for name, kwargs, status, controls, needles in cases:
            expect(
                name,
                run(_synthetic_stage_dict(repo, **kwargs), controls=controls),
                status,
                *needles,
            )

        # 2. hole 2: an unrelated crash, and a red run at the wrong step, are not
        #    controls even though the command exits nonzero.
        expect(
            "reject_unrelated_crash",
            run(
                _synthetic_stage_dict(
                    repo,
                    steps=[["python3", "crash.py"]],
                    old="return 1;",
                    new="boom",
                    expected_output="synthetic body change rejected",
                )
            ),
            "fail",
            "expected rejecting result",
        )
        expect(
            "reject_wrong_red_step",
            run(
                _synthetic_stage_dict(
                    repo,
                    steps=[["python3", "crash.py"], ["python3", "check.py"]],
                    old="return 1;",
                    new="boom",
                    expected_step=1,
                    expected_output="synthetic body change rejected",
                )
            ),
            "fail",
            "control target not reached",
        )

        # 3. the mutation engine fails loudly on an absent or ambiguous target.
        try:
            Mutation(path="x", old="absent", new="y").apply("alpha\nbeta\n")
            record("mutation_absent_raises", False, "missing target did not raise")
        except MutationError as exc:
            record("mutation_absent_raises", True, f"raised: {exc}")
        try:
            Mutation(path="x", old="a", new="b").apply("a\na\n")
            record("mutation_ambiguous_raises", False, "ambiguous target did not raise")
        except MutationError as exc:
            record("mutation_ambiguous_raises", True, f"raised: {exc}")

        # 4. hole 4: a symlinked ancestor of the mutation target is refused and the
        #    real source is left untouched.
        before = (repo / "song" / "synthetic.slang").read_bytes()
        escape = run(_synthetic_stage_dict(repo, link=["song"]), controls=True)
        after = (repo / "song" / "synthetic.slang").read_bytes()
        record(
            "reject_symlink_ancestor_escape",
            escape.status == "blocked"
            and "outside the resolved throwaway root" in escape.reason
            and before == after,
            f"status={escape.status} reason={escape.reason}",
        )

        # 5. hole 3: single-root resolution. The explicit Song root wins over an
        #    ordinary sibling checkout, and there is no sibling fallback.
        shadow = workdir / "shadow"
        explicit = shadow / "repo" / "song"
        sibling = shadow / "song"
        explicit.mkdir(parents=True, exist_ok=True)
        sibling.mkdir(parents=True, exist_ok=True)
        (explicit / "synthetic.slang").write_text("explicit\n", encoding="utf-8")
        (sibling / "synthetic.slang").write_text("sibling\n", encoding="utf-8")
        won = resolve_one("song/synthetic.slang", shadow / "repo", explicit)
        record(
            "explicit_song_root_wins",
            won is not None
            and won.resolve() == (explicit / "synthetic.slang").resolve(),
            f"resolved={won}",
        )
        (explicit / "synthetic.slang").unlink()
        fallback = resolve_one("song/synthetic.slang", shadow / "repo", explicit)
        record(
            "explicit_song_root_no_sibling_fallback",
            fallback is None,
            f"resolved={fallback}",
        )

        # 6. the manifest rejects a stale or missing selected source.
        for name, path, needle in (
            ("reject_manifest_stale", "song/synthetic.slang", "stale source input"),
            ("reject_manifest_missing", "song/absent.slang", "missing source input"),
        ):
            manifest = (Artifact(kind=SOURCE_ARTIFACT, path=path, sha256="0" * 64),)
            expect(
                name,
                run(_synthetic_stage_dict(repo), controls=False, manifest=manifest),
                "blocked",
                needle,
            )

        # 7. controls.json parses, declares the eight stages and the expected
        #    control status of each.
        try:
            loaded = load_controls()
            ids = {stage.id for stage in loaded.stages}
            expected = {
                "decode",
                "machine",
                "source",
                "init",
                "lookup",
                "write",
                "resources",
                "replacement-link",
            }
            record(
                "controls_schema",
                ids == expected and len(loaded.stages) == 8,
                f"ids={sorted(ids)}",
            )
            by_id = {stage.id: stage for stage in loaded.stages}
            for name, root in loaded.roots.items():
                record(
                    f"controls_root:{name}",
                    (REPO_ROOT / root).exists() or (REPO_ROOT.parent / root).exists(),
                    f"root={root}",
                )
            for name in ("lookup",):
                stage = by_id[name]
                record(
                    f"controls_awaiting:{name}",
                    stage.control_status == AWAITING
                    and stage.mutation is None
                    and bool(stage.awaiting_reason),
                    f"status={stage.control_status}",
                )
            write = by_id["write"].mutation
            record(
                "controls_anchor:write",
                write is not None
                and write.anchor is not None
                and "oldVal" in write.new
                and "else value" in write.old,
                f"anchor={write.anchor if write else None}",
            )
            for name, step, needle in (
                ("decode", 1, "statement count differs"),
                ("machine", 0, "EvalExpr.conditionalTrue"),
            ):
                control = by_id[name].mutation
                record(
                    f"controls_expect:{name}",
                    control is not None
                    and control.expected_step == step
                    and needle in control.expected_output,
                    f"step={control.expected_step if control else None}",
                )
            record(
                "controls_manifest",
                len(loaded.manifest) >= 8
                and all(entry.kind == SOURCE_ARTIFACT for entry in loaded.manifest),
                f"entries={len(loaded.manifest)}",
            )
        except ControlsError as exc:
            record("controls_schema", False, str(exc))

        # 8. every declared active mutation resolves uniquely against its real source.
        try:
            loaded = load_controls()
        except ControlsError:
            loaded = Controls(roots={}, manifest=(), stages=())
        for stage in loaded.stages:
            if stage.mutation is None:
                continue
            resolved = resolve_one(stage.mutation.path, REPO_ROOT, DEFAULT_SONG_ROOT)
            if resolved is None:
                record(
                    f"mutation_resolves:{stage.id}",
                    False,
                    f"missing source {stage.mutation.path}",
                )
                continue
            try:
                pristine = resolved.read_text(encoding="utf-8")
                mutated = stage.mutation.apply(pristine)
                record(
                    f"mutation_resolves:{stage.id}",
                    mutated != pristine,
                    "target resolves uniquely and changes the source",
                )
            except (MutationError, OSError) as exc:
                record(f"mutation_resolves:{stage.id}", False, str(exc))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    for check in checks:
        print(json.dumps(check, sort_keys=True))
    failed = [check for check in checks if not check["ok"]]
    print(
        json.dumps(
            {"self_test": "pass" if not failed else "fail", "checks": len(checks)},
            sort_keys=True,
        )
    )
    return EXIT_OK if not failed else EXIT_BLOCKED

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _select(stages: tuple[Stage, ...], requested: list[str]) -> list[Stage]:
    if not requested:
        return []
    wanted: list[str] = []
    for item in requested:
        wanted.extend(part for part in item.split(",") if part)
    by_id = {stage.id: stage for stage in stages}
    unknown = [name for name in wanted if name not in by_id]
    if unknown:
        raise SystemExit(f"unknown stage id(s): {', '.join(unknown)}")
    return [by_id[name] for name in wanted]

def cmd_list() -> int:
    try:
        controls = load_controls()
    except ControlsError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    print("manifest:")
    for entry in controls.manifest:
        print(f"    source {entry.path} sha256={entry.sha256}")
    for stage in sorted(controls.stages, key=lambda item: item.order):
        print(
            f"{stage.order:>2}  {stage.id:<16} entry={stage.source_entry:<3} "
            f"control={stage.rejecting_control:<3} [{stage.control_status}] {stage.title}"
        )
        print(f"    {stage.summary}")
        for artifact in stage.artifacts:
            if artifact.kind == PROOF_ARTIFACT:
                target = artifact.declaration or f"{artifact.declaration_prefix}.*"
                print(f"    proof  {artifact.root}:{artifact.module}::{target}")
            else:
                print(f"    source {artifact.path} sha256={artifact.sha256}")
        steps = " ; ".join(" ".join(step) for step in stage.command.steps)
        print(f"    command ({stage.command.cwd}) {steps}")
        if stage.mutation is not None:
            print(
                f"    mutation {stage.mutation.path} step={stage.mutation.expected_step} "
                f"expect={stage.mutation.expect}"
            )
            print(f"      expects output: {stage.mutation.expected_output!r}")
        else:
            print(f"    control awaiting definition: {stage.awaiting_reason}")
        if stage.note:
            print(f"    note: {stage.note}")
    return EXIT_OK

def cmd_run(args: argparse.Namespace) -> int:
    try:
        controls = load_controls()
    except ControlsError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    try:
        selected = _select(controls.stages, args.stage)
    except SystemExit as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not selected:
        print("FAIL: no stage selected", file=sys.stderr)
        return EXIT_USAGE
    song_root = Path(args.song_root).resolve()
    if not song_root.is_dir():
        print(f"FAIL: --song-root does not resolve: {song_root}", file=sys.stderr)
        return EXIT_BLOCKED
    throwaway_root = Path(args.throwaway_root)
    throwaway_root.mkdir(parents=True, exist_ok=True)

    results = [
        check_stage(
            stage,
            controls.roots,
            controls.manifest,
            REPO_ROOT,
            song_root,
            controls=args.controls,
            throwaway_root=throwaway_root,
            keep=args.keep,
            timeout_override=args.timeout,
        )
        for stage in sorted(selected, key=lambda item: item.order)
    ]
    for result in results:
        print(result.to_json())
    failures = [result for result in results if result.status != "pass"]
    summary = {
        "selected": len(results),
        "passed": len(results) - len(failures),
        "blocked_or_failed": len(failures),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    if failures:
        reasons = "; ".join(f"{r.stage}: {r.reason}" for r in failures)
        print(f"FAIL: {reasons}", file=sys.stderr)
        return EXIT_BLOCKED
    return EXIT_OK

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="refine.py",
        description="Staged Slang source-refinement gate harness.",
    )
    parser.add_argument("--list", action="store_true", help="list the declared stages")
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="ID",
        help="stage to run (repeatable, comma-separated)",
    )
    parser.add_argument(
        "--controls",
        action="store_true",
        help="run the baseline / mutation / restored-copy control",
    )
    parser.add_argument("--self-test", action="store_true", help="verify the harness logic")
    parser.add_argument(
        "--song-root",
        default=str(DEFAULT_SONG_ROOT),
        help=f"Song checkout root (default: {DEFAULT_SONG_ROOT})",
    )
    parser.add_argument(
        "--throwaway-root",
        default=str(DEFAULT_THROWAWAY_ROOT),
        help=f"root for throwaway copies (default: {DEFAULT_THROWAWAY_ROOT})",
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="override the per-command timeout"
    )
    parser.add_argument("--keep", action="store_true", help="keep the throwaway copy")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.list:
        return cmd_list()
    if not args.stage:
        parser.error("one of --list, --self-test or --stage is required")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    return cmd_run(args)

if __name__ == "__main__":
    sys.exit(main())
