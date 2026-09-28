#!/usr/bin/env python3
"""Rejecting zero-axiom runner for the Slang source-refinement chain.

For one registered stage this runner builds and elaborates the stage's audit
module and requires that every registered declaration reports an **empty**
`Lean.collectAxioms` set. The Lean side is `proof/corpus/Slang/Audit/Empty.lean`
and the registry is `tools/slang/zero_axioms.json`; this runner is the command
that fails on a nonempty set, a missing name, a wrong type, an unregistered
declaration or a missing stage.

What is checked
===============
* every registered declaration logs exactly one `EMPTY_AXIOMS_OK <name>` line;
  an extra name is an unregistered declaration, a missing one is unaudited;
* the audit module's inline `#audit_empty <name> : <type>` lines match the
  registry's names and expected types, so the registry is the authoritative
  registered type rather than a comment;
* registered child-import closures extend that same source scan to every reached
  child module; cycles, missing modules and paths outside the registered scope reject;
* the stage's source files contain no `sorry`, `admit`, `native_decide`,
  `noncomputable` or `classical` token (comments and strings are stripped).
  This is **auxiliary**: Lean does not fail a build for a merely-unused
  `noncomputable`, so this scan is a source-level belt, not the acceptance test;
* with `--controls`, the baseline is green, a mutation of the real fixture is
  red and the restored copy is green, and every rejection control below is
  demonstrated red.

Rejection controls
==================
* a throwaway declaration whose proof uses `Classical.choice`, `funext`
  (which reaches `Quot.sound`) or `Quot.sound`;
* a `sorry`, `admit` or `native_decide` declaration (the last reports the
  generated `_native` auxiliary axiom);
* a wrong registered type, a missing declaration and an `axiom` target;
* a green declaration not in the registry (coverage rejection);
* a missing stage and a placeholder stage (nonzero exit);
* a source file carrying a forbidden token.

Isolation
=========
Every Lean command runs under `systemd-run --user --scope -p MemoryMax=6G
-p MemorySwapMax=0` with `LEAN_NUM_THREADS=1` and a `timeout`, holding one
serialized lock for the shared `proof/corpus/.lake/` lane. `ulimit -v` is never
used. Throwaway control files are written under the system temporary directory;
the repository working tree is never mutated.

Usage::

    python3 tools/slang/zero_axioms.py --list
    python3 tools/slang/zero_axioms.py --stage harness
    python3 tools/slang/zero_axioms.py --stage harness --controls

Exit status: 0 when the stage passes; 1 when it is blocked or a control does not
reject; 2 on a usage error; 3 when a required tool is missing.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
CORPUS = REPO / "proof/corpus"
REGISTRY_PATH = HERE.parent / "zero_axioms.json"
WORK_DIR = Path(tempfile.gettempdir()) / "opencode" / "zero-axioms"

EXIT_OK = 0
EXIT_REJECT = 1
EXIT_USAGE = 2
EXIT_TOOL = 3

OK_LINE = re.compile(r"^EMPTY_AXIOMS_OK\s+(\S+)\s*$", re.MULTILINE)
AUDIT_LINE = re.compile(r"^\s*#audit_empty\s+([A-Za-z0-9_.'?✝]+)\s*:\s*(.+?)\s*$")
LEAN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)+")


class GateError(Exception):
    """The stage cannot reach its acceptance condition."""


class RegistryError(GateError):
    """The registry is missing or malformed."""


class StageError(GateError):
    """The requested stage is not registered."""


class PlaceholderError(GateError):
    """The requested stage is a placeholder with no registered declarations."""


class AuditError(GateError):
    """The audit command did not run or did not exit clean."""


class CoverageError(GateError):
    """The reported audit population does not match the registry."""


class TokenError(GateError):
    """A registry source file carries a forbidden token."""


class ControlError(GateError):
    """A rejection control did not reject."""


class ToolError(GateError):
    """A required external tool is missing."""


@dataclass(frozen=True)
class Declaration:
    name: str
    expected_type: str
    module: str
    path: str


@dataclass(frozen=True)
class SourceImportClosure:
    root: str
    allowed_imports: tuple[str, ...]


@dataclass(frozen=True)
class StageSpec:
    id: str
    kind: str
    wave: str
    title: str
    note: str
    audit_module: str | None
    source_files: tuple[str, ...]
    declarations: tuple[Declaration, ...]
    source_import_closures: tuple[SourceImportClosure, ...] = ()


@dataclass(frozen=True)
class Registry:
    memory_max: str
    memory_swap_max: str
    timeout: int
    forbidden_tokens: tuple[str, ...]
    stages: dict[str, StageSpec]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def _string(value: object, where: str) -> str:
    _require(isinstance(value, str) and value.strip() != "", f"{where} must be a nonempty string")
    return value  # type: ignore[return-value]


def load_registry(path: Path = REGISTRY_PATH) -> Registry:
    if not path.is_file():
        raise RegistryError(f"registry missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"registry is not valid JSON: {exc}") from exc
    _require(isinstance(data, dict), "registry must be an object")
    _require(data.get("schema") == 1, "registry schema must be 1")
    defaults = data.get("defaults", {})
    _require(isinstance(defaults, dict), "defaults must be an object")
    memory_max = _string(defaults.get("memory_max", "6G"), "defaults.memory_max")
    memory_swap_max = _string(defaults.get("memory_swap_max", "0"), "defaults.memory_swap_max")
    timeout = defaults.get("timeout_seconds", 900)
    _require(isinstance(timeout, int) and timeout > 0, "defaults.timeout_seconds must be positive")
    tokens_raw = data.get("forbidden_tokens", [])
    _require(isinstance(tokens_raw, list) and tokens_raw, "forbidden_tokens must be a nonempty list")
    tokens = tuple(_string(token, "forbidden_tokens[]") for token in tokens_raw)
    stages_raw = data.get("stages")
    _require(isinstance(stages_raw, dict) and stages_raw, "stages must be a nonempty object")

    stages: dict[str, StageSpec] = {}
    for stage_id, raw in stages_raw.items():
        where = f"stages.{stage_id}"
        _require(isinstance(raw, dict), f"{where} must be an object")
        kind = _string(raw.get("kind"), f"{where}.kind")
        _require(kind in ("active", "placeholder"), f"{where}.kind must be active or placeholder")
        wave = _string(raw.get("wave"), f"{where}.wave")
        title = _string(raw.get("title"), f"{where}.title")
        note = str(raw.get("note", ""))
        declarations_raw = raw.get("declarations")
        _require(isinstance(declarations_raw, list), f"{where}.declarations must be a list")
        declarations: list[Declaration] = []
        for index, item in enumerate(declarations_raw):
            dw = f"{where}.declarations[{index}]"
            _require(isinstance(item, dict), f"{dw} must be an object")
            declarations.append(
                Declaration(
                    name=_string(item.get("name"), f"{dw}.name"),
                    expected_type=_string(item.get("expected_type"), f"{dw}.expected_type"),
                    module=_string(item.get("module"), f"{dw}.module"),
                    path=_string(item.get("path"), f"{dw}.path"),
                )
            )
        if kind == "active":
            audit_module = _string(raw.get("audit_module"), f"{where}.audit_module")
            source_raw = raw.get("source_files")
            _require(isinstance(source_raw, list) and source_raw, f"{where}.source_files must be a nonempty list")
            source_files = tuple(_string(entry, f"{where}.source_files[]") for entry in source_raw)
        else:
            _require(not declarations, f"{where} is a placeholder and must carry no declarations")
            audit_module = None
            source_files = ()
        closures_raw = raw.get("source_import_closures", [])
        _require(isinstance(closures_raw, list), f"{where}.source_import_closures must be a list")
        _require(kind == "active" or not closures_raw, f"{where}: placeholder has an import closure")
        closures: list[SourceImportClosure] = []
        for index, scope in enumerate(closures_raw):
            sw = f"{where}.source_import_closures[{index}]"
            _require(isinstance(scope, dict) and set(scope) == {"root", "allowed_imports"},
                     f"{sw} requires root and allowed_imports only")
            root = _string(scope.get("root"), f"{sw}.root")
            _require(LEAN_NAME.fullmatch(root) is not None, f"{sw}.root is not a module name")
            root_path = "proof/corpus/" + root.replace(".", "/") + ".lean"
            _require(root_path in source_files, f"{sw}.root must be an explicit source file")
            allowed_raw = scope.get("allowed_imports")
            _require(isinstance(allowed_raw, list), f"{sw}.allowed_imports must be a list")
            allowed = tuple(_string(name, f"{sw}.allowed_imports[]") for name in allowed_raw)
            _require(all(LEAN_NAME.fullmatch(name) is not None for name in allowed),
                     f"{sw}.allowed_imports contains an invalid module name")
            _require(len(set(allowed)) == len(allowed), f"{sw}.allowed_imports has duplicates")
            _require(all(name != root and not name.startswith(root + ".") for name in allowed),
                     f"{sw}.allowed_imports must not exempt closure members")
            closures.append(SourceImportClosure(root, allowed))
        _require(len({scope.root for scope in closures}) == len(closures),
                 f"{where} has duplicate import-closure roots")
        names = [decl.name for decl in declarations]
        _require(len(names) == len(set(names)), f"{where} has duplicate declaration name(s)")
        stages[stage_id] = StageSpec(
            id=stage_id,
            kind=kind,
            wave=wave,
            title=title,
            note=note,
            audit_module=audit_module,
            source_files=source_files,
            declarations=tuple(declarations),
            source_import_closures=tuple(closures),
        )
    return Registry(
        memory_max=memory_max,
        memory_swap_max=memory_swap_max,
        timeout=timeout,
        forbidden_tokens=tokens,
        stages=stages,
    )


def select_stage(registry: Registry, stage_id: str) -> StageSpec:
    if stage_id not in registry.stages:
        registered = ", ".join(sorted(registry.stages))
        raise StageError(f"missing stage {stage_id!r}; registered stages: {registered}")
    stage = registry.stages[stage_id]
    if stage.kind != "active":
        raise PlaceholderError(
            f"stage {stage_id!r} is a placeholder with no registered declarations; "
            "populate it before it can gate"
        )
    return stage


# ---------------------------------------------------------------------------
# Coverage, type binding and source tokens
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_ok(output: str) -> list[str]:
    return OK_LINE.findall(output or "")


def parse_audit_module(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = AUDIT_LINE.match(line)
        if match is not None:
            found[match.group(1)] = normalize(match.group(2))
    return found


def check_coverage(reported: list[str], expected: set[str]) -> None:
    duplicates = sorted({name for name in reported if reported.count(name) > 1})
    if duplicates:
        raise CoverageError(f"duplicate audit report(s): {duplicates}")
    reported_set = set(reported)
    extra = sorted(reported_set - expected)
    if extra:
        raise CoverageError(f"unregistered declaration(s) reported: {extra}")
    missing = sorted(expected - reported_set)
    if missing:
        raise CoverageError(f"registered declaration(s) not audited: {missing}")


def check_type_binding(stage: StageSpec, module_text: str) -> None:
    inline = parse_audit_module(module_text)
    registered = {decl.name: normalize(decl.expected_type) for decl in stage.declarations}
    extra = sorted(set(inline) - set(registered))
    if extra:
        raise CoverageError(f"audit module declares unregistered name(s): {extra}")
    missing = sorted(set(registered) - set(inline))
    if missing:
        raise CoverageError(f"audit module omits registered name(s): {missing}")
    for name in sorted(registered):
        if inline[name] != registered[name]:
            raise CoverageError(
                f"registered type mismatch for {name}: registry {registered[name]!r} "
                f"!= audit module {inline[name]!r}"
            )


def strip_lean(text: str) -> str:
    """Remove nested Lean block comments, line comments and string literals."""
    out: list[str] = []
    index = 0
    depth = 0
    size = len(text)
    while index < size:
        char = text[index]
        if depth == 0 and char == '"':
            out.append(char)
            index += 1
            while index < size:
                inner = text[index]
                out.append(inner)
                if inner == "\\" and index + 1 < size:
                    out.append(text[index + 1])
                    index += 2
                    continue
                index += 1
                if inner == '"':
                    break
            continue
        if char == "/" and index + 1 < size and text[index + 1] == "-":
            depth += 1
            out.append("  ")
            index += 2
            continue
        if depth > 0 and char == "-" and index + 1 < size and text[index + 1] == "/":
            depth -= 1
            out.append("  ")
            index += 2
            continue
        if depth > 0:
            out.append("\n" if char == "\n" else " ")
            index += 1
            continue
        if char == "-" and index + 1 < size and text[index + 1] == "-":
            while index < size and text[index] != "\n":
                index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def check_source_tokens(path: Path, tokens: tuple[str, ...]) -> None:
    if not path.is_file():
        raise TokenError(f"registry source file missing: {path}")
    code = strip_lean(path.read_text(encoding="utf-8"))
    for token in tokens:
        pattern = rf"(?<![A-Za-z0-9_']){re.escape(token)}(?![A-Za-z0-9_'])"
        if re.search(pattern, code, re.IGNORECASE):
            raise TokenError(f"forbidden token {token!r} in {path}")


# ---------------------------------------------------------------------------
# Isolated Lean execution
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def lane_lock():
    """Serialize every Lean run on the shared proof/corpus/.lake lane."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(REPO).encode("utf-8")).hexdigest()[:16]
    lock_path = WORK_DIR / f"lane-{digest}.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def lean_argv(registry: Registry, target: str, timeout: int) -> list[str]:
    if shutil.which("systemd-run") is None:
        raise ToolError("systemd-run is required for the cgroup lane; it is not on PATH")
    inner = [
        "timeout",
        str(timeout),
        "env",
        "LEAN_NUM_THREADS=1",
        "lake",
        "env",
        "lean",
        target,
    ]
    return [
        "systemd-run",
        "--user",
        "--scope",
        "-p",
        f"MemoryMax={registry.memory_max}",
        "-p",
        f"MemorySwapMax={registry.memory_swap_max}",
        "--",
        *inner,
    ]


def run_lean(registry: Registry, target: str, timeout: int | None = None) -> tuple[int, str]:
    seconds = timeout or registry.timeout
    argv = lean_argv(registry, target, seconds)
    environment = dict(os.environ)
    environment["LEAN_NUM_THREADS"] = "1"
    try:
        completed = subprocess.run(
            argv,
            cwd=str(CORPUS),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=seconds + 120,
        )
    except subprocess.TimeoutExpired as exc:
        raise AuditError(f"timeout after {seconds}s running {target}") from exc
    except OSError as exc:
        raise ToolError(f"subprocess failed to start ({target}): {exc}") from exc
    return completed.returncode, completed.stdout or ""


def tail(text: str, limit: int = 500) -> str:
    return (text or "").strip()[-limit:]


def failure_lines(output: str) -> str:
    """The error lines of a merged Lean run, which precede the buffered info logs."""
    lines = [line for line in (output or "").splitlines() if "error" in line.lower()]
    return " | ".join(lines[-3:]) if lines else tail(output)


def source_imports(path: Path) -> list[str]:
    """Read generated Lean's simple import header; O(bytes) time and space.

    Comment stripping visits each byte once; header lines and module tokens are
    then visited once. Unsupported header forms reject rather than omit imports.
    """
    imports: list[str] = []
    for line in strip_lean(path.read_text(encoding="utf-8")).splitlines():
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if words[0] == "import":
            if len(words) == 1 or any(LEAN_NAME.fullmatch(name) is None for name in words[1:]):
                raise TokenError(f"invalid source import in {path}: {line}")
            imports.extend(words[1:])
        elif words[0] in {"prelude", "public", "meta", "private", "protected"}:
            # Generated files have ordinary imports followed by namespace/definitions.
            # A private definition ends the header; a qualified import is unsupported.
            if words[0] == "prelude" or "import" in words:
                raise TokenError(f"unsupported source import header in {path}: {line}")
            break
        elif words[0] in {"namespace", "section", "open", "set_option", "def", "theorem",
                          "lemma", "abbrev", "example", "instance", "structure", "inductive", "class"}:
            break
        else:
            raise TokenError(f"unsupported source import header in {path}: {line}")
    return imports


def check_source_closure(scope: SourceImportClosure, registry: Registry,
                         corpus: Path) -> tuple[Path, ...]:
    """Scan each child module once, with active/done DFS states.

    O(B + E) time and O(B + V + E) space for B source bytes, V modules and E
    imports: each file is read/scanned once, and each import edge is visited once.
    """
    corpus = corpus.resolve()
    active: set[str] = set()
    done: set[str] = set()
    allowed = set(scope.allowed_imports)
    scanned: list[Path] = []
    # Enter/leave events avoid a Python recursion limit on a long valid chain.
    pending = [(scope.root, False)]
    while pending:
        name, leaving = pending.pop()
        if leaving:
            active.remove(name)
            done.add(name)
            continue
        if name in active:
            raise TokenError(f"source import cycle at {name}")
        if name in done:
            continue
        path = corpus.joinpath(*name.split(".")).with_suffix(".lean")
        resolved = path.resolve()
        if resolved != path or not resolved.is_relative_to(corpus):
            raise TokenError(f"escaped source import path for {name}: {path}")
        if not path.is_file():
            raise TokenError(f"source import missing: {name}: {path}")
        check_source_tokens(path, registry.forbidden_tokens)
        imports = source_imports(path)
        children: list[str] = []
        for imported in imports:
            if imported == scope.root or imported.startswith(scope.root + "."):
                children.append(imported)
            elif imported in allowed:
                dependency = corpus.joinpath(*imported.split(".")).with_suffix(".lean")
                if dependency.resolve() != dependency or not dependency.is_file():
                    raise TokenError(f"missing or escaped permitted source import: {imported}")
            else:
                raise TokenError(f"source import outside registered closure: {imported}")
        scanned.append(path)
        active.add(name)
        pending.append((name, True))
        pending.extend((child, False) for child in reversed(children))
    return tuple(scanned)


def check_stage_sources(stage: StageSpec, registry: Registry) -> tuple[Path, ...]:
    """Scan explicit sources and registered children in O(total bytes + edges).

    Existing explicit sources remain included; closure traversal adds every
    imported child without changing declaration/type coverage.
    """
    scanned: list[Path] = []
    for rel in stage.source_files:
        path = REPO / rel
        check_source_tokens(path, registry.forbidden_tokens)
        scanned.append(path)
    for scope in stage.source_import_closures:
        scanned.extend(check_source_closure(scope, registry, CORPUS))
    return tuple(scanned)


# ---------------------------------------------------------------------------
# Stage audit
# ---------------------------------------------------------------------------

def audit_module_path(stage: StageSpec) -> Path:
    assert stage.audit_module is not None
    return CORPUS / stage.audit_module


def run_baseline(stage: StageSpec, registry: Registry) -> list[str]:
    path = audit_module_path(stage)
    if not path.is_file():
        raise AuditError(f"audit module missing: {path}")
    expected = {decl.name for decl in stage.declarations}
    if not expected:
        raise AuditError(f"stage {stage.id!r} has no registered declarations")
    check_stage_sources(stage, registry)
    check_type_binding(stage, path.read_text(encoding="utf-8"))
    returncode, output = run_lean(registry, stage.audit_module)
    if returncode != 0:
        raise AuditError(
            f"audit exited {returncode} (a nonempty axiom set, wrong type or missing "
            f"declaration is a failure): {failure_lines(output)}"
        )
    reported = parse_ok(output)
    check_coverage(reported, expected)
    print(f"BASELINE GREEN: {len(reported)}/{len(expected)} declarations report an empty axiom set")
    return reported


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

MUTATION_OLD = "theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl"
MUTATION_NEW = "theorem nat_add_zero : (0 : Nat) + 0 = 0 := Classical.choice ⟨rfl⟩"

REJECTION_CONTROLS: tuple[tuple[str, str, str], ...] = (
    (
        "throwaway-choice",
        "set_option warningAsError false\n"
        "theorem cChoice : (0 : Nat) = 0 := Classical.choice ⟨rfl⟩\n"
        "#audit_empty cChoice : (0 : Nat) = 0",
        "Classical.choice",
    ),
    (
        "throwaway-propext",
        "theorem cPropext (a b : Prop) (h : a ↔ b) : a = b := propext h\n"
        "#audit_empty cPropext : ∀ (a b : Prop), (a ↔ b) → a = b",
        "propext",
    ),
    (
        "transitive-helper",
        "set_option warningAsError false\n"
        "theorem cHelper : (0 : Nat) = 0 := Classical.choice ⟨rfl⟩\n"
        "theorem cTop : (0 : Nat) = 0 := cHelper\n"
        "#audit_empty cTop : (0 : Nat) = 0",
        "Classical.choice",
    ),
    (
        "instance-dependency",
        "set_option warningAsError false\n"
        "class C (α : Type) where out : α\n"
        "noncomputable instance instCNat : C Nat := ⟨Classical.choice ⟨0⟩⟩\n"
        "theorem cInst : (C.out : Nat) = (C.out : Nat) := rfl\n"
        "#audit_empty cInst : (C.out : Nat) = (C.out : Nat)",
        "Classical.choice",
    ),
    (
        "throwaway-funext",
        "theorem cFunext : (fun x : Nat => x) = (fun x : Nat => x) := by funext x; rfl\n"
        "#audit_empty cFunext : (fun x : Nat => x) = (fun x : Nat => x)",
        "Quot.sound",
    ),
    (
        "throwaway-quot",
        "theorem cQuot : Quot.mk (fun _ _ : Nat => True) 0 = "
        "Quot.mk (fun _ _ : Nat => True) 1 := Quot.sound trivial\n"
        "#audit_empty cQuot : Quot.mk (fun _ _ : Nat => True) 0 = "
        "Quot.mk (fun _ _ : Nat => True) 1",
        "Quot.sound",
    ),
    (
        "throwaway-sorry",
        "set_option warningAsError false\n"
        "theorem cSorry : False := by sorry\n"
        "#audit_empty cSorry : False",
        "sorryAx",
    ),
    (
        "throwaway-admit",
        "set_option warningAsError false\n"
        "theorem cAdmit : False := by admit\n"
        "#audit_empty cAdmit : False",
        "sorryAx",
    ),
    (
        "throwaway-native-decide",
        "theorem cNative : (1 + 1 = 2) := by native_decide\n"
        "#audit_empty cNative : (1 + 1 = 2)",
        "native",
    ),
    (
        "wrong-type",
        "theorem cType : (1 : Nat) = 1 := rfl\n"
        "#audit_empty cType : (0 : Nat) = 0",
        "does not have the registered type",
    ),
    (
        "missing-declaration",
        "#audit_empty cMissing : (0 : Nat) = 0",
        "declaration not found",
    ),
    (
        "axiom-target",
        "axiom cAxiom : (0 : Nat) = 0\n"
        "#audit_empty cAxiom : (0 : Nat) = 0",
        "is an axiom",
    ),
)


def write_control(name: str, text: str) -> Path:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    path = WORK_DIR / f"control_{name}.lean"
    path.write_text(text, encoding="utf-8")
    return path


def expect_red(name: str, path: Path, registry: Registry, needle: str) -> None:
    returncode, output = run_lean(registry, str(path))
    if returncode == 0:
        raise ControlError(f"control {name} survived: audit exited 0, expected rejection")
    if needle not in output:
        raise ControlError(f"control {name} failed without the expected {needle!r}: {failure_lines(output)}")
    print(f"CONTROL {name}: RED exit={returncode} :: {failure_lines(output)}")


def run_controls(stage: StageSpec, registry: Registry, pristine: str) -> None:
    expected = {decl.name for decl in stage.declarations}

    # baseline / red / restored-green on the real audit module
    if MUTATION_OLD not in pristine:
        raise ControlError(f"mutation anchor absent from {stage.audit_module}")
    mutant = write_control("mutation", pristine.replace(MUTATION_OLD, MUTATION_NEW, 1))
    expect_red("mutation-classical-choice", mutant, registry, "Classical.choice")
    restored = write_control("mutation", pristine)
    returncode, output = run_lean(registry, str(restored))
    if returncode != 0:
        raise ControlError(f"restored copy failed (exit {returncode}): {failure_lines(output)}")
    check_coverage(parse_ok(output), expected)
    print("CONTROL restored-green: GREEN")

    # throwaway declarations that must make the command throw
    for name, extra, needle in REJECTION_CONTROLS:
        expect_red(name, write_control(name, pristine + "\n" + extra + "\n"), registry, needle)

    # a green declaration outside the registry must fail the coverage check
    extra = "theorem cExtra : (0 : Nat) = 0 := rfl\n#audit_empty cExtra : (0 : Nat) = 0\n"
    path = write_control("unregistered", pristine + "\n" + extra)
    returncode, output = run_lean(registry, str(path))
    if returncode != 0:
        raise ControlError(f"unregistered control audit failed unexpectedly: {failure_lines(output)}")
    try:
        check_coverage(parse_ok(output), expected)
    except CoverageError as exc:
        print(f"CONTROL unregistered-declaration: RED :: {exc}")
    else:
        raise ControlError("unregistered declaration survived the coverage check")

    # a registered declaration absent from the report must fail the coverage check
    reduced = sorted(expected)[:-1]
    try:
        check_coverage(reduced, expected)
    except CoverageError as exc:
        print(f"CONTROL missing-declaration-report: RED :: {exc}")
    else:
        raise ControlError("missing declaration survived the coverage check")

    # a missing stage and a placeholder stage must be rejected
    try:
        select_stage(registry, "no-such-stage")
    except StageError as exc:
        print(f"CONTROL missing-stage: RED :: {exc}")
    else:
        raise ControlError("missing stage survived")

    # a placeholder stage must be rejected; the stage id is selected dynamically
    # (or synthesized) so activating a real stage cannot silently disable this control
    placeholder_id = next(
        (sid for sid, spec in registry.stages.items() if spec.kind != "active"), None
    )
    if placeholder_id is None:
        placeholder_id = "__placeholder_control__"
        placeholder_registry = Registry(
            memory_max=registry.memory_max,
            memory_swap_max=registry.memory_swap_max,
            timeout=registry.timeout,
            forbidden_tokens=registry.forbidden_tokens,
            stages={
                placeholder_id: StageSpec(
                    id=placeholder_id,
                    kind="placeholder",
                    wave="control",
                    title="placeholder control",
                    note="synthetic placeholder used only to exercise select_stage",
                    audit_module=None,
                    source_files=(),
                    declarations=(),
                )
            },
        )
    else:
        placeholder_registry = registry

    try:
        select_stage(placeholder_registry, placeholder_id)
    except PlaceholderError as exc:
        print(f"CONTROL placeholder-stage: RED :: {exc}")
    else:
        raise ControlError("placeholder stage survived")

    # the auxiliary source-token scan must itself be able to fail on every token
    for token in registry.forbidden_tokens:
        poisoned = write_control(
            "tokens",
            pristine + "\ntheorem cToken : (0 : Nat) = 0 := by\n  " + token + "\n",
        )
        try:
            check_source_tokens(poisoned, registry.forbidden_tokens)
        except TokenError as exc:
            print(f"CONTROL forbidden-token {token!r}: RED :: {exc}")
        else:
            raise ControlError(f"forbidden token {token!r} survived the source scan")

    print(
        "CONTROLS PASS: baseline green, mutation red, restored green, "
        f"{len(REJECTION_CONTROLS) + 5} rejection controls red"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_list(registry: Registry) -> int:
    for stage_id in sorted(registry.stages):
        stage = registry.stages[stage_id]
        marker = "active" if stage.kind == "active" else "placeholder"
        print(f"{stage_id:<16} [{marker}] wave={stage.wave} {stage.title}")
        if stage.audit_module:
            print(f"    audit module: {stage.audit_module}")
        for decl in stage.declarations:
            print(f"    {decl.name} :: {decl.expected_type}")
    return EXIT_OK


def run_stage(stage_id: str, controls: bool) -> int:
    registry = load_registry()
    stage = select_stage(registry, stage_id)
    with lane_lock():
        run_baseline(stage, registry)
        if controls:
            run_controls(stage, registry, audit_module_path(stage).read_text(encoding="utf-8"))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zero_axioms.py",
        description="Rejecting zero-axiom runner for the Slang source-refinement chain.",
    )
    parser.add_argument("--stage", help="registered stage to audit")
    parser.add_argument("--controls", action="store_true", help="run the rejection controls")
    parser.add_argument("--list", action="store_true", help="list the registry")
    args = parser.parse_args(argv)

    if args.list:
        try:
            return cmd_list(load_registry())
        except GateError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return EXIT_REJECT
    if not args.stage:
        parser.error("--stage or --list is required")
    try:
        return run_stage(args.stage, args.controls)
    except ToolError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_TOOL
    except GateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_REJECT


if __name__ == "__main__":
    sys.exit(main())
