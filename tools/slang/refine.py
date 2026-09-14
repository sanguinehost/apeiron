#!/usr/bin/env python3
"""Fail-closed staged Slang source-refinement gate harness (APEIRON-PROGRAMME-W21-T1).

The harness reads the stage table in ``tools/slang/controls.json`` and, for one
selected stage, checks the stage's own artifacts and runs its acceptance
command. With ``--controls`` it additionally runs a baseline / mutation /
restored-copy control in a throwaway copy of the tree, so the repository working
tree is never mutated.

Fail-closed contract
====================
Acceptance for a stage requires every one of the following; each failure is a
red result with a named reason and a nonzero exit status:

* every declared artifact resolves -- a missing proof module, a missing
  fully-qualified declaration under its namespace, or a missing source input is
  a blocked stage, never a pass;
* every declared source input digest matches the working-tree bytes -- a stale
  digest is a blocked stage;
* the stage command starts and exits 0 -- a timeout or a nonzero exit is a
  blocked stage;
* with ``--controls``, the declared mutation of the actual body turns the stage
  command red, and the restored copy turns it green again -- a surviving
  mutation or a failing restored copy is a failed stage.

The harness establishes gate consistency only. It does not prove a refinement,
does not establish a source-semantics link, and never reports correspondence.
A green stage means the declared artifacts and control behave as recorded, not
that the represented program is correct.

Stages are data. The eight stage ids are the W21-T1 harness names; each binds
to one wave of ``docs/plan/slang-source-refinement.md``: ``decode`` -> W22,
``machine`` -> W23, ``source`` -> W24, ``init`` -> W25, ``lookup`` -> W26,
``write`` -> W27, ``resources`` -> W28, ``replacement-link`` -> W29. The proof
module, command and negative control are the plan's declared outputs for the
future ``proof/corpus/Slang/*`` and ``proof/corpus/Joint/*`` tasks; they are
recorded here so a missing module, generator, checker or binding is a named red
result. The encoded mutation is the plan's negative control expressed as an
exact replacement against the real Song source. Where the plan names a decoded
Lean artifact, the stage command must re-render the binding from the mutated
source; a command that ignores it reports ``mutation survived`` (red), never a
pass.

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
DEFAULT_THROWAWAY_ROOT = Path("/tmp/opencode/slang-refine")
DEFAULT_TIMEOUT = 1800

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_USAGE = 2
EXIT_TOOL = 3

HEX64 = re.compile(r"[0-9a-f]{64}")
DECL_LINE = re.compile(
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+|partial\s+|unsafe\s+)*"
    r"(theorem|lemma|def|abbrev|structure|class|inductive|instance)\s+"
    r"([A-Za-z_][A-Za-z0-9_.']*)"
)
OPEN_NAMESPACE = re.compile(r"^\s*namespace\s+([\w.]+)")
CLOSE_NAMESPACE = re.compile(r"^\s*end\b\s*([\w.]*)")


class GateError(Exception):
    """A stage could not reach its acceptance condition."""


class ArtifactError(GateError):
    """A declared artifact is missing or stale."""


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
    mutation: Mutation
    copy: tuple[str, ...]
    link: tuple[str, ...]
    note: str = ""


@dataclasses.dataclass
class StageResult:
    stage: str
    status: str
    reason: str
    baseline: str = "not-run"
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


def _variants(path: str) -> list[str]:
    variants = [path]
    if path.startswith("song/"):
        variants.append(path[len("song/") :])
    return variants


def resolve_many(path: str, repo_root: Path, song_root: Path) -> list[Path]:
    """Resolve a declared path under the repository and Song checkouts.

    A ``song/``-prefixed path is tried both as given and with the prefix
    stripped, so the recorded W18 paths resolve against ``--song-root``.
    Globs are expanded; a literal path must exist to resolve.
    """
    matches: list[Path] = []
    for base in (repo_root, repo_root.parent, song_root):
        for variant in _variants(path):
            if any(ch in variant for ch in "*?["):
                try:
                    matches.extend(sorted(base.glob(variant)))
                except (OSError, ValueError):
                    continue
            else:
                candidate = base / variant
                if candidate.exists():
                    matches.append(candidate)
    seen: set[Path] = set()
    unique: list[Path] = []
    for match in matches:
        resolved = match.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(match)
    return unique


def resolve_one(path: str, repo_root: Path, song_root: Path) -> Path | None:
    matches = resolve_many(path, repo_root, song_root)
    return matches[0] if matches else None


def lean_declarations(text: str) -> dict[str, set[str]]:
    """Map every fully-qualified Lean declaration name to its keywords.

    The namespace stack is tracked exactly, so a declaration binds to its real
    enclosing namespace rather than to a prefix that happens to be opened.
    """
    declarations: dict[str, set[str]] = {}
    stack: list[str] = []
    for line in text.splitlines():
        match = DECL_LINE.match(line)
        if match is not None:
            keyword, name = match.group(1), match.group(2)
            full = ".".join(stack + [name]) if stack else name
            declarations.setdefault(full, set()).add(keyword)
        opened = OPEN_NAMESPACE.match(line)
        if opened is not None:
            stack.extend(opened.group(1).split("."))
            continue
        closed = CLOSE_NAMESPACE.match(line)
        if closed is not None:
            name = closed.group(1)
            if name:
                for _ in name.split("."):
                    if stack:
                        stack.pop()
            elif stack:
                stack.pop()
    return declarations


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
    if kind == "proof":
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
    if kind == "source":
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
    return Mutation(
        path=path,
        old=old,
        new=new,
        anchor=anchor,
        expect=str(raw.get("expect", "")),
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
        if artifact.kind == "proof":
            _require(
                artifact.root in roots,
                f"{where}: proof root {artifact.root!r} is not declared in `roots`",
            )
    command = _parse_command(raw.get("command"), f"{where}.command", default_timeout)
    mutation = _parse_mutation(raw.get("mutation"), f"{where}.mutation")
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
        note=str(raw.get("note", "")),
    )


def parse_controls(data: object) -> tuple[dict[str, str], list[Stage]]:
    _require(isinstance(data, dict), "controls.json must be an object")
    _require(data.get("schema") == 1, "controls.json schema must be 1")
    roots_raw = data.get("roots")
    _require(isinstance(roots_raw, dict) and roots_raw, "controls.json `roots` missing")
    roots: dict[str, str] = {}
    for name, value in roots_raw.items():
        roots[_require_str(name, "roots key")] = _require_str(value, f"roots.{name}")
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
    return roots, stages


def load_controls(path: Path = CONTROLS_PATH) -> tuple[dict[str, str], list[Stage]]:
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


def check_artifact(
    artifact: Artifact, roots: dict[str, str], repo_root: Path, song_root: Path
) -> None:
    if artifact.kind == "proof":
        base = repo_root / roots[artifact.root]  # type: ignore[index]
        module = base / artifact.module  # type: ignore[arg-type]
        if not module.is_file():
            raise ArtifactError(
                f"missing proof target: {artifact.root}:{artifact.module}"
            )
        try:
            text = module.read_text(encoding="utf-8")
        except OSError as exc:
            raise ArtifactError(
                f"proof target unreadable: {artifact.root}:{artifact.module} ({exc})"
            ) from exc
        declarations = lean_declarations(text)
        if artifact.declaration is not None and artifact.declaration not in declarations:
            raise ArtifactError(
                f"missing proof target declaration {artifact.declaration} in "
                f"{artifact.root}:{artifact.module}"
            )
        if artifact.declaration_prefix is not None:
            prefix = artifact.declaration_prefix
            if not any(
                name == prefix or name.startswith(prefix + ".")
                for name in declarations
            ):
                raise ArtifactError(
                    f"missing proof target under namespace {prefix} in "
                    f"{artifact.root}:{artifact.module}"
                )
        return
    if artifact.kind == "source":
        resolved = resolve_one(artifact.path, repo_root, song_root)  # type: ignore[arg-type]
        if resolved is None:
            raise ArtifactError(f"missing source input: {artifact.path}")
        actual = sha256_file(resolved)
        if actual != artifact.sha256:
            raise ArtifactError(
                f"stale source input: {artifact.path} digest {actual} != recorded "
                f"{artifact.sha256}"
            )
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


def build_throwaway(
    stage: Stage, dest: Path, repo_root: Path, song_root: Path
) -> None:
    """Copy the declared inputs into ``dest``; symlink the declared read-only trees.

    The repository is only read. A missing real-copy input is a hard error; a
    missing link target is skipped, so an absent optional cache is not a false
    red.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for entry in stage.copy:
        matches = resolve_many(entry, repo_root, song_root)
        if not matches:
            raise GateError(f"throwaway copy input missing: {entry}")
        globbed = any(ch in entry for ch in "*?[")
        parent = Path(entry).parent if globbed else None
        for match in matches:
            target = (dest / parent / match.name) if globbed else (dest / entry)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if match.is_dir():
                shutil.copytree(match, target)
            else:
                shutil.copy2(match, target)
    for entry in stage.link:
        matches = resolve_many(entry, repo_root, song_root)
        if not matches:
            continue
        match = matches[0]
        target = dest / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        os.symlink(match.resolve(), target)

    # The mutation target must be a real copy in the throwaway, even when the
    # stage's copy manifest does not list it; a symlink would write through to
    # the repository.
    mutation_target = dest / stage.mutation.path
    if not mutation_target.exists() or mutation_target.is_symlink():
        mutation_source = resolve_one(stage.mutation.path, repo_root, song_root)
        if mutation_source is None:
            raise GateError(f"mutation source missing: {stage.mutation.path}")
        mutation_target.parent.mkdir(parents=True, exist_ok=True)
        if mutation_target.is_symlink():
            mutation_target.unlink()
        shutil.copy2(mutation_source, mutation_target)


def _expand(argv: tuple[str, ...], dest: Path, repo_root: Path, song_root: Path) -> list[str]:
    return [
        token.replace("{song_root}", str(song_root))
        .replace("{repo}", str(repo_root))
        .replace("{root}", str(dest))
        for token in argv
    ]


def run_steps(
    stage: Stage,
    dest: Path,
    repo_root: Path,
    song_root: Path,
    timeout_override: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the stage steps in order; return the first nonzero result, else the last.

    A timeout or an unstartable executable raises a named :class:`GateError`.
    """
    timeout = timeout_override or stage.command.timeout
    cwd = dest / stage.command.cwd
    result: subprocess.CompletedProcess[str] | None = None
    for argv in stage.command.steps:
        expanded = _expand(argv, dest, repo_root, song_root)
        try:
            result = subprocess.run(
                expanded,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise CommandTimeout(
                f"timeout after {timeout}s running {' '.join(expanded)}"
            ) from exc
        except OSError as exc:
            raise CommandFailure(
                f"subprocess failed to start ({' '.join(expanded)}): {exc}"
            ) from exc
        if result.returncode != 0:
            return result
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text[-limit:]


def check_stage(
    stage: Stage,
    roots: dict[str, str],
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

    mutated_path = dest / stage.mutation.path
    try:
        pristine = mutated_path.read_bytes()
    except OSError as exc:
        result.reason = f"mutation target missing in throwaway: {stage.mutation.path} ({exc})"
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
            f"{_tail(baseline.stdout)}"
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
        mutated = stage.mutation.apply(pristine.decode("utf-8"))
    except (MutationError, UnicodeDecodeError) as exc:
        result.status = "fail"
        result.reason = f"mutation rejected: {exc}"
        _cleanup(dest, keep)
        return result
    try:
        mutated_path.write_text(mutated, encoding="utf-8")
    except OSError as exc:
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
    result.red_exit = red.returncode
    if red.returncode == 0:
        result.status = "fail"
        result.reason = (
            "mutation survived: command exited 0 (expected red at "
            f"{stage.mutation.expect or 'the declared mutation'})"
        )
        _cleanup(dest, keep)
        return result

    try:
        mutated_path.write_bytes(pristine)
    except OSError as exc:
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
            f"restored copy failed (exit {green.returncode}): {_tail(green.stdout)}"
        )
        _cleanup(dest, keep)
        return result

    result.status = "pass"
    result.reason = (
        f"baseline green, mutation red (exit {red.returncode}), restored green"
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
    (root / "song" / "synthetic.slang").write_text("body { return 1; }\n", encoding="utf-8")
    (root / "check.py").write_text(
        "import pathlib, sys\n"
        "text = pathlib.Path('song/synthetic.slang').read_text()\n"
        "sys.exit(2 if 'return 2;' in text else 0)\n",
        encoding="utf-8",
    )
    return root


def _synthetic_stage_dict(
    root: Path,
    *,
    module: str = "Slang/Synthetic.lean",
    prefix: str = "Slang.Synthetic",
    digest: str | None = None,
    steps: list[list[str]] | None = None,
    timeout: int = 60,
    old: str = "return 1;",
    new: str = "return 2;",
) -> dict:
    if digest is None:
        digest = sha256_file(root / "song" / "synthetic.slang")
    return {
        "id": "synthetic",
        "order": 1,
        "title": "synthetic stage",
        "summary": "self-test synthetic stage",
        "source_entry": "SE0",
        "rejecting_control": "SC0",
        "artifacts": [
            {
                "kind": "proof",
                "root": "bridge",
                "module": module,
                "declaration_prefix": prefix,
            },
            {"kind": "source", "path": "song/synthetic.slang", "sha256": digest},
        ],
        "command": {
            "cwd": ".",
            "timeout": timeout,
            "steps": steps or [["python3", "check.py"]],
        },
        "mutation": {
            "path": "song/synthetic.slang",
            "old": old,
            "new": new,
            "expect": "synthetic body change rejected",
        },
        "copy": ["song/synthetic.slang", "Slang/Synthetic.lean", "check.py"],
        "link": [],
    }


def self_test() -> int:
    """Prove the rejection classes fire and the positive accepts."""
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    workdir = Path(tempfile.mkdtemp(prefix="slang-refine-self-", dir="/tmp/opencode"))
    try:
        repo = _synthetic_repo(workdir / "repo")
        roots = {"bridge": "."}

        def stage_from(raw: dict) -> Stage:
            return parse_stage(
                raw,
                "synthetic",
                roots=roots,
                default_timeout=60,
                default_copy=(),
                default_link=(),
            )

        def run(raw: dict, *, controls: bool = True) -> StageResult:
            parsed = stage_from(raw)
            return check_stage(
                parsed,
                roots,
                repo,
                repo,
                controls=controls,
                throwaway_root=workdir / "throwaway",
            )

        # 0. the positive control: present-and-fresh artifact, green baseline,
        #    mutation red then restored green.
        positive = run(_synthetic_stage_dict(repo), controls=True)
        record(
            "positive_present_and_fresh",
            positive.status == "pass" and positive.red_exit not in (None, 0)
            and positive.green_exit == 0,
            f"status={positive.status} reason={positive.reason}",
        )

        # 1. missing artifact: a proof module that does not exist.
        missing = run(
            _synthetic_stage_dict(repo, module="Slang/Absent.lean"), controls=False
        )
        record(
            "reject_missing_artifact",
            missing.status == "blocked" and "missing proof target" in missing.reason,
            f"status={missing.status} reason={missing.reason}",
        )

        # 1b. missing artifact: a declared declaration absent from a present module.
        missing_decl = run(
            _synthetic_stage_dict(repo, prefix="Slang.Other"), controls=False
        )
        record(
            "reject_missing_declaration",
            missing_decl.status == "blocked"
            and "missing proof target under namespace" in missing_decl.reason,
            f"status={missing_decl.status} reason={missing_decl.reason}",
        )

        # 2. stale digest: a recorded sha256 that does not match the bytes.
        stale = run(
            _synthetic_stage_dict(repo, digest="0" * 64), controls=False
        )
        record(
            "reject_stale_digest",
            stale.status == "blocked" and "stale source input" in stale.reason,
            f"status={stale.status} reason={stale.reason}",
        )

        # 3. timeout: a command that outlives its declared timeout.
        timed = run(
            _synthetic_stage_dict(
                repo,
                steps=[["python3", "-c", "import time; time.sleep(30)"]],
                timeout=1,
            ),
            controls=False,
        )
        record(
            "reject_timeout",
            timed.status == "blocked" and "timeout" in timed.reason,
            f"status={timed.status} reason={timed.reason}",
        )

        # 4. nonzero subprocess: a command that exits nonzero in the baseline.
        nonzero = run(
            _synthetic_stage_dict(repo, steps=[["python3", "-c", "import sys; sys.exit(7)"]]),
            controls=False,
        )
        record(
            "reject_nonzero_subprocess",
            nonzero.status == "blocked"
            and "nonzero subprocess" in nonzero.reason
            and "exit 7" in nonzero.reason,
            f"status={nonzero.status} reason={nonzero.reason}",
        )

        # 5. an unstartable executable is a named red, not a crash.
        unstartable = run(
            _synthetic_stage_dict(repo, steps=[["__apeiron_absent_tool__"]]),
            controls=False,
        )
        record(
            "reject_unstartable_subprocess",
            unstartable.status == "blocked"
            and "failed to start" in unstartable.reason,
            f"status={unstartable.status} reason={unstartable.reason}",
        )

        # 6. a surviving mutation is rejected: the command still exits 0.
        # The mutation edits text the synthetic checker ignores, so the red run
        # is green and the harness must reject the control.
        survived = run(
            _synthetic_stage_dict(repo, old="body {", new="body  {"), controls=True
        )
        record(
            "reject_surviving_mutation",
            survived.status == "fail" and "mutation survived" in survived.reason,
            f"status={survived.status} reason={survived.reason}",
        )

        # 7. the mutation engine fails loudly on an absent or ambiguous target.
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

        # 8. controls.json parses and declares the eight required stages.
        try:
            parsed_roots, stages = load_controls()
            ids = {stage.id for stage in stages}
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
                ids == expected and len(stages) == 8,
                f"ids={sorted(ids)}",
            )
            for name, root in parsed_roots.items():
                record(
                    f"controls_root:{name}",
                    (REPO_ROOT / root).exists() or (REPO_ROOT.parent / root).exists(),
                    f"root={root}",
                )
        except ControlsError as exc:
            record("controls_schema", False, str(exc))

        # 9. every declared mutation resolves uniquely against its real source.
        try:
            _, stages = load_controls()
        except ControlsError:
            stages = []
        for stage in stages:
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


def _select(stages: list[Stage], requested: list[str]) -> list[Stage]:
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
        _, stages = load_controls()
    except ControlsError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    for stage in sorted(stages, key=lambda item: item.order):
        print(
            f"{stage.order:>2}  {stage.id:<16} entry={stage.source_entry:<3} "
            f"control={stage.rejecting_control:<3} {stage.title}"
        )
        print(f"    {stage.summary}")
        for artifact in stage.artifacts:
            if artifact.kind == "proof":
                target = artifact.declaration or f"{artifact.declaration_prefix}.*"
                print(f"    proof  {artifact.root}:{artifact.module}::{target}")
            else:
                print(f"    source {artifact.path} sha256={artifact.sha256}")
        steps = " ; ".join(" ".join(step) for step in stage.command.steps)
        print(f"    command ({stage.command.cwd}) {steps}")
        print(f"    mutation {stage.mutation.path}: {stage.mutation.expect}")
        if stage.note:
            print(f"    note: {stage.note}")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    try:
        roots, stages = load_controls()
    except ControlsError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    try:
        selected = _select(stages, args.stage)
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
            roots,
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
    parser.add_argument("--timeout", type=int, default=None, help="override the per-command timeout")
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
