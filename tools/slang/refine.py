#!/usr/bin/env python3
"""Fail-closed staged Slang source-refinement gate harness (APEIRON-PROGRAMME-W21-T2).

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
``docs/contracts/slang-source-refinement-plan.md``.

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

from proof_target import ProofTargetError, verify as verify_proof_target

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
CONTROLS_PATH = HERE.parent / "controls.json"
DEFAULT_SONG_ROOT = REPO_ROOT.parent / "song"
DEFAULT_THROWAWAY_ROOT = Path("/tmp/opencode/slang-refine")
DEFAULT_TIMEOUT = 1800

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_USAGE = 2
EXIT_TOOL = 3

HEX64 = re.compile(r"[0-9a-f]{64}")
STAGE_ID = re.compile(r"[A-Za-z0-9._-]+")
UNBOUND = "unbound"
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
    declaration_type: str | None = None
    declaration_prefix: str | None = None
    path: str | None = None
    sha256: str | None = None
    note: str = ""

    @property
    def unbound(self) -> bool:
        """A namespace-prefix target with no exact declaration or type binding."""
        return self.kind == PROOF_ARTIFACT and self.declaration is None

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
    canonical_bindings: list[str] = dataclasses.field(default_factory=list)

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
        expected = raw.get("declaration_type")
        prefix = raw.get("declaration_prefix")
        binding = raw.get("binding", "exact")
        _require(
            binding in ("exact", UNBOUND),
            f"{where}.binding must be 'exact' or {UNBOUND!r}",
        )
        if declaration is not None:
            _require_str(declaration, f"{where}.declaration")
            _require(
                expected is not None,
                f"{where}: exact proof target requires `declaration_type`",
            )
            _require_str(expected, f"{where}.declaration_type")
            _require(
                prefix is None,
                f"{where}: `declaration_prefix` is forbidden with an exact `declaration`",
            )
            _require(
                binding == "exact",
                f"{where}: an exact `declaration` cannot be marked {UNBOUND!r}",
            )
        else:
            _require_str(prefix, f"{where}.declaration_prefix")
            _require(
                binding == UNBOUND,
                f"{where}: a namespace-prefix target must declare "
                f"`binding`: {UNBOUND!r}",
            )
            _require(
                expected is None,
                f"{where}: {UNBOUND!r} target cannot declare `declaration_type`",
            )
        return Artifact(
            kind=kind,
            root=root,
            module=module,
            declaration=declaration,
            declaration_type=expected,
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

def check_proof_declarations(
    stage: Stage, roots: dict[str, str], dest: Path, timeout: int
) -> list[str]:
    """Build each module under test, then bind its exact declaration and type.

    A declaration that exists only as commented text, a same-named helper in the
    namespace, or a declaration with the wrong proposition is rejected by the
    generated probe; the empty axiom cone is checked in the same probe.
    """
    resolved_dest = dest.resolve()
    canonical_bindings: list[str] = []
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
        try:
            canonical_bindings.extend(
                verify_proof_target(
                    package_dir,
                    module_rel,
                    artifact.declaration,  # type: ignore[arg-type]
                    artifact.declaration_type or "",
                    resolved_dest,
                    timeout,
                )
            )
        except ProofTargetError as exc:
            raise ArtifactError(
                f"{exc} [{artifact.root}:{module_rel}::{artifact.declaration}]"
            ) from exc
    return sorted(set(canonical_bindings))

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
    unbound = [a.declaration_prefix for a in stage.artifacts if a.unbound]
    if unbound:
        result.reason = f"unbound proof headline: {unbound}"
        return result
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
        result.canonical_bindings = check_proof_declarations(
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

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> int:
    """Delegate to the self-test module to keep this harness within the size bound."""
    import refine_selftest
    return refine_selftest.self_test()

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
