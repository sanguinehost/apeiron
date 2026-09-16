#!/usr/bin/env python3
"""Rejecting gate for the W24 generated `DecodeTrace.Run` derivations.

The W24 generation barrier emits untrusted generated `Run` witnesses for every
recursive call and continuation of the decoder. This runner is the missing
proof-producing check mode for that output: an untrusted producer (later
`tools/slang/export.py`) records a manifest of the bytes it emitted, the
derivation nodes it produced and the operations it used; this runner rejects the
manifest unless every recorded digest is fresh, every expected request
constructor and lemma is present, no unsupported producer operation appears, and
the generated `Run` witnesses kernel-check with an empty axiom cone.

What is checked
===============
* stale bytes -- every recorded artifact digest is recomputed from the file on
  disk; a missing file or a digest mismatch is rejected before any Lean run;
* missing node -- the report carries one node for every request constructor and
  every registered lemma of the phase, drawn from the `derivation` stage of
  `tools/slang/zero_axioms.json`; an absent expected node is rejected;
* unsupported operation -- the report's producer vocabulary must be a subset of
  the registered `ask`/`done`/`bind`/... set, and every request node must name a
  constructor of the phase; anything else is rejected;
* false-green kernel target -- a generated probe module imports the emitted
  witness module and, for each lemma node, requires the declaration to exist, not
  be an axiom, have head `...Run`/`...Derives`, be definitionally equal to the
  recorded type, and carry an empty axiom cone; a run that exits 0 without
  confirming a witness is a false green;
* inconclusive -- a timeout or an OOM kill is reported as inconclusive with a
  nonzero exit, never as a pass.

Phases are data: `grammar`, `statement`, `selection`, `validation`, `decoder`
and `whole-source`, each bound to one derivation module and the exact declaration
names registered for that module. No generator is required to register them;
`--phase` fails closed while the producer report is absent.

Isolation
=========
The real kernel check writes its probe under the system temporary directory,
compiles the generated module onto the shared `proof/corpus/.lake/` lane with
`lake env lean -o <olean>` (the generated modules are not Lake roots, so
`lake build <module>` reports `unknown target`), then runs `lake env lean` on the
probe. Both steps run under
`systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0` with
`LEAN_NUM_THREADS=1` and a `timeout`, holding one serialized lock for the shared
lane. `ulimit -v` is never used. The repository source tree is never mutated (the
`.olean` lands under `.lake/build/`, the normal build artifact directory);
`--self-test` and the synthetic controls need no Lean.

Usage::

    python3 tools/slang/trace_check.py --list
    python3 tools/slang/trace_check.py --phase decoder
    python3 tools/slang/trace_check.py --controls
    python3 tools/slang/trace_check.py --self-test

Exit status: 0 when the phase passes; 1 when it is blocked or a control does not
reject; 2 on a usage error; 3 when a required tool is missing.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
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
REPORT_DIR = CORPUS / "Slang/Generated/Trace"
WORK_DIR = Path(tempfile.gettempdir()) / "opencode" / "trace-check"
DEFAULT_SONG_ROOT = REPO.parent / "song"
DEFAULT_TIMEOUT = 1800

EXIT_OK = 0
EXIT_REJECT = 1
EXIT_USAGE = 2
EXIT_TOOL = 3

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
LEAN_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*")
OK_LINE = re.compile(r"^TRACE_OK\s+(\S+)\s*$", re.MULTILINE)
OOM_CODES = (137, -9)

# The `Layer`/`Derives` primitives a generated derivation may use. The producer
# task extends this set only by editing this registry, not by emitting silently.
ALLOWED_OPERATIONS = ("ask", "done", "bind", "ite", "match", "rfl", "decide", "guard")

# Phase id -> derivation module. The module's declaration names come from the
# `derivation` stage of zero_axioms.json; the request constructors below are the
# `inductive Request where` constructors of that module.
PHASE_MODULES: dict[str, str] = {
    "grammar": "Slang.Derivation.Grammar",
    "statement": "Slang.Derivation.Statements",
    "selection": "Slang.Derivation.Selection",
    "validation": "Slang.Derivation.Validation",
    "decoder": "Slang.Derivation.Decode",
    "whole-source": "Slang.Derivation.Composition",
}
PHASE_ORDER = tuple(PHASE_MODULES)
PHASE_TITLES: dict[str, str] = {
    "grammar": "Expression grammar derivations",
    "statement": "Statement derivations",
    "selection": "Declaration selection derivations",
    "validation": "Validation derivations",
    "decoder": "Decoder call-and-continuation derivations",
    "whole-source": "Whole-source composition derivations",
}
# Request constructors, read from the module source at registration time:
# proof/corpus/Slang/Derivation/<Module>.lean. `Composition` has no Request.
PHASE_REQUESTS: dict[str, tuple[str, ...]] = {
    "grammar": ("expr", "primary", "args", "binary"),
    "statement": ("block", "stmt", "expr"),
    "selection": ("parameters", "line", "function", "scan", "expr", "block"),
    "validation": (
        "delimiters",
        "distinct",
        "spans",
        "expressionCalls",
        "expressionsCalls",
        "statementCalls",
        "statementsCalls",
        "resolveCalls",
        "expressionNames",
        "expressionsNames",
        "statementsNames",
        "resolveNames",
    ),
    "decoder": (
        "utf8Fuel",
        "lexing",
        "selection",
        "validation",
        "decodeTokens",
        "decodeTable",
        "importedCoord",
        "closure",
    ),
    "whole-source": (),
}


class GateError(Exception):
    """The phase cannot reach its acceptance condition."""


class RegistryError(GateError):
    """The derivation registry is missing or malformed."""


class PhaseError(GateError):
    """The requested phase is not registered."""


class ProducerAbsentError(GateError):
    """The untrusted producer has not emitted the phase report."""


class ReportError(GateError):
    """The producer report is missing or malformed."""


class StaleError(GateError):
    """A recorded artifact digest does not match the bytes on disk."""


class MissingNodeError(GateError):
    """An expected request constructor or lemma is absent from the report."""


class UnsupportedOperationError(GateError):
    """The report uses an unregistered producer operation or request constructor."""


class WitnessTypeError(GateError):
    """A recorded lemma node does not name a `Run`/`Derives` proposition."""


class KernelError(GateError):
    """The generated witnesses did not kernel-check."""


class FalseGreenError(GateError):
    """The kernel run exited 0 without confirming a recorded witness."""


class InconclusiveError(GateError):
    """A timeout or OOM makes the kernel result inconclusive, not a pass."""


class ControlError(GateError):
    """A control did not reject, or a restored copy did not pass."""


class ToolError(GateError):
    """A required external tool is missing."""


@dataclass(frozen=True)
class PhaseSpec:
    id: str
    title: str
    module: str
    path: str
    requests: tuple[str, ...]
    declarations: tuple[str, ...]


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str
    kind: str  # "generated" | "bytes" | "source"


@dataclass(frozen=True)
class Node:
    kind: str  # "request" | "lemma"
    name: str
    witness: str = ""
    expected_type: str = ""


@dataclass(frozen=True)
class Report:
    schema: int
    phase: str
    song_revision: str
    artifacts: tuple[Artifact, ...]
    nodes: tuple[Node, ...]
    operations: tuple[str, ...]


# ---------------------------------------------------------------------------
# Phase registry (data only; no generator required)
# ---------------------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def _nonempty(value: object, where: str) -> str:
    _require(isinstance(value, str) and value.strip() != "", f"{where} must be a nonempty string")
    return value  # type: ignore[return-value]


def load_phases(path: Path = REGISTRY_PATH) -> dict[str, PhaseSpec]:
    """Register the six phases from the `derivation` stage of the registry.

    The module and declaration names are read from `zero_axioms.json`; a missing
    module or a duplicate declaration name is a registry error, not a pass.
    """
    if not path.is_file():
        raise RegistryError(f"registry missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"registry is not valid JSON: {exc}") from exc
    _require(isinstance(data, dict), "registry must be an object")
    stages = data.get("stages")
    _require(isinstance(stages, dict), "registry `stages` must be an object")
    stage = stages.get("derivation")
    _require(isinstance(stage, dict), "registry has no `derivation` stage")
    declarations = stage.get("declarations")
    _require(isinstance(declarations, list) and declarations, "derivation stage has no declarations")

    by_module: dict[str, list[dict]] = {}
    for index, item in enumerate(declarations):
        where = f"derivation.declarations[{index}]"
        _require(isinstance(item, dict), f"{where} must be an object")
        module = _nonempty(item.get("module"), f"{where}.module")
        _nonempty(item.get("name"), f"{where}.name")
        _nonempty(item.get("path"), f"{where}.path")
        by_module.setdefault(module, []).append(item)

    phases: dict[str, PhaseSpec] = {}
    for phase_id, module in PHASE_MODULES.items():
        entries = by_module.get(module)
        if not entries:
            raise RegistryError(
                f"derivation stage registers no declaration for module {module} (phase {phase_id})"
            )
        names = [entry["name"] for entry in entries]
        _require(len(names) == len(set(names)), f"phase {phase_id} has duplicate declaration names")
        phases[phase_id] = PhaseSpec(
            id=phase_id,
            title=PHASE_TITLES[phase_id],
            module=module,
            path=entries[0]["path"],
            requests=PHASE_REQUESTS[phase_id],
            declarations=tuple(names),
        )
    return phases


def select_phase(phases: dict[str, PhaseSpec], phase_id: str) -> PhaseSpec:
    if phase_id not in phases:
        registered = ", ".join(PHASE_ORDER)
        raise PhaseError(f"missing phase {phase_id!r}; registered phases: {registered}")
    return phases[phase_id]


def report_path_for(phase_id: str) -> Path:
    return REPORT_DIR / f"{phase_id}.json"


# ---------------------------------------------------------------------------
# Report parsing and pure checks
# ---------------------------------------------------------------------------

def _require_report(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def _report_str(value: object, where: str) -> str:
    _require_report(
        isinstance(value, str) and value.strip() != "",
        f"{where} must be a nonempty string",
    )
    return value  # type: ignore[return-value]


def parse_report(data: object, where: str = "report") -> Report:
    _require_report(isinstance(data, dict), f"{where} must be an object")
    _require_report(data.get("schema") == 1, f"{where}.schema must be 1")
    phase = _report_str(data.get("phase"), f"{where}.phase")
    revision = _report_str(data.get("song_revision"), f"{where}.song_revision")
    _require_report(HEX40.fullmatch(revision) is not None, f"{where}.song_revision must be 40 hex")

    artifacts_raw = data.get("artifacts")
    _require_report(
        isinstance(artifacts_raw, list) and artifacts_raw,
        f"{where}.artifacts must be a nonempty list",
    )
    artifacts: list[Artifact] = []
    for index, item in enumerate(artifacts_raw):
        aw = f"{where}.artifacts[{index}]"
        _require_report(isinstance(item, dict), f"{aw} must be an object")
        artifact_path = _report_str(item.get("path"), f"{aw}.path")
        digest = _report_str(item.get("sha256"), f"{aw}.sha256")
        _require_report(HEX64.fullmatch(digest) is not None, f"{aw}.sha256 must be a 64-hex digest")
        kind = _report_str(item.get("kind"), f"{aw}.kind")
        _require_report(
            kind in ("generated", "bytes", "source"),
            f"{aw}.kind must be generated/bytes/source",
        )
        artifacts.append(Artifact(path=artifact_path, sha256=digest, kind=kind))
    _require_report(
        any(artifact.kind == "generated" for artifact in artifacts),
        f"{where} must record at least one generated module",
    )

    nodes_raw = data.get("nodes")
    _require_report(isinstance(nodes_raw, list), f"{where}.nodes must be a list")
    nodes: list[Node] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(nodes_raw):
        nw = f"{where}.nodes[{index}]"
        _require_report(isinstance(item, dict), f"{nw} must be an object")
        kind = _report_str(item.get("kind"), f"{nw}.kind")
        _require_report(kind in ("request", "lemma"), f"{nw}.kind must be request or lemma")
        name = _report_str(item.get("name"), f"{nw}.name")
        key = (kind, name)
        _require_report(key not in seen, f"{where} has duplicate node {key}")
        seen.add(key)
        if kind == "request":
            # A request node is the generated `Run` witness for one recursive call.
            witness = _report_str(item.get("witness"), f"{nw}.witness")
            expected = _report_str(item.get("type"), f"{nw}.type")
        elif item.get("witness") is not None or item.get("type") is not None:
            # A lemma node may carry a witness, but must then carry both fields.
            witness = _report_str(item.get("witness"), f"{nw}.witness")
            expected = _report_str(item.get("type"), f"{nw}.type")
        else:
            witness = ""
            expected = ""
        nodes.append(Node(kind=kind, name=name, witness=witness, expected_type=expected))

    operations_raw = data.get("operations")
    _require_report(isinstance(operations_raw, list), f"{where}.operations must be a list")
    operations = tuple(_report_str(op, f"{where}.operations[]") for op in operations_raw)
    return Report(
        schema=1,
        phase=phase,
        song_revision=revision,
        artifacts=tuple(artifacts),
        nodes=tuple(nodes),
        operations=operations,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_artifact(path: str, repo_root: Path, song_root: Path) -> Path:
    """Resolve a recorded artifact under exactly one root; reject escapes."""
    if path == "song" or path.startswith("song/"):
        base = song_root
        relative = path[4:] if path == "song" else path[len("song/") :]
    else:
        base = repo_root
        relative = path
    resolved_base = base.resolve()
    candidate = (base / relative).resolve()
    try:
        common = os.path.commonpath([str(candidate), str(resolved_base)])
    except ValueError as exc:
        raise StaleError(f"artifact path escapes its root: {path}") from exc
    if common != str(resolved_base):
        raise StaleError(f"artifact path escapes its root: {path}")
    return candidate


def check_stale(report: Report, repo_root: Path, song_root: Path) -> None:
    """Reject a missing generated artifact or a digest that is not fresh."""
    for artifact in report.artifacts:
        target = resolve_artifact(artifact.path, repo_root, song_root)
        if not target.is_file():
            raise StaleError(f"generated artifact missing: {artifact.path} ({target})")
        actual = sha256_file(target)
        if actual != artifact.sha256:
            raise StaleError(
                f"stale generated bytes: {artifact.path} digest {actual} != recorded {artifact.sha256}"
            )


def check_nodes(report: Report, spec: PhaseSpec) -> None:
    """Reject a report missing an expected request constructor or lemma node."""
    reported_requests = {node.name for node in report.nodes if node.kind == "request"}
    reported_lemmas = {node.name for node in report.nodes if node.kind == "lemma"}
    missing_requests = sorted(set(spec.requests) - reported_requests)
    if missing_requests:
        raise MissingNodeError(f"phase {spec.id}: missing request node(s): {missing_requests}")
    missing_lemmas = sorted(set(spec.declarations) - reported_lemmas)
    if missing_lemmas:
        raise MissingNodeError(f"phase {spec.id}: missing lemma node(s): {missing_lemmas}")


def check_operations(report: Report, spec: PhaseSpec) -> None:
    """Reject an unregistered producer operation or request constructor."""
    unsupported = sorted(set(report.operations) - set(ALLOWED_OPERATIONS))
    if unsupported:
        raise UnsupportedOperationError(
            f"phase {spec.id}: unsupported producer operation(s): {unsupported}"
        )
    reported_requests = {node.name for node in report.nodes if node.kind == "request"}
    unknown = sorted(reported_requests - set(spec.requests))
    if unknown:
        raise UnsupportedOperationError(
            f"phase {spec.id}: unknown request constructor(s): {unknown}"
        )


def check_witness_types(report: Report, spec: PhaseSpec) -> None:
    """Reject a request node whose recorded type is not a `Run`/`Derives` claim.

    Request nodes are the generated derivations for recursive calls, so each must
    name a `Run` proposition. Lemma nodes are registered declarations that may be
    structural (`evaluate`, `body_eq`), so they are not required to be `Run`.
    """
    for node in report.nodes:
        if node.kind != "request":
            continue
        if "Run" not in node.expected_type and "Derives" not in node.expected_type:
            raise WitnessTypeError(
                f"phase {spec.id}: request node {node.name} declares a non-Run type: "
                f"{node.expected_type!r}"
            )


# ---------------------------------------------------------------------------
# Kernel probe and isolated Lean execution
# ---------------------------------------------------------------------------

def lean_string(value: str) -> str:
    return json.dumps(value)


def kernel_probe_text(import_name: str, nodes: tuple[Node, ...]) -> str:
    """Generate the probe that binds each recorded `Run` witness in the kernel."""
    pairs = ",\n    ".join(
        f"({lean_string(node.witness)}, {lean_string(node.expected_type)})" for node in nodes
    )
    return (
        f"import {import_name}\n"
        "import Lean\n"
        "open Lean Elab Command Meta\n"
        "run_cmd do\n"
        "  let env \u2190 getEnv\n"
        "  let targets : List (String \u00d7 String) := [\n"
        f"    {pairs}\n"
        "  ]\n"
        "  for target in targets do\n"
        "    let witness := target.1\n"
        "    let expectedSrc := target.2\n"
        "    let some info := env.find? witness.toName\n"
        "      | throwError s!\"missing trace witness: {witness}\"\n"
        "    if let .axiomInfo _ := info then\n"
        "      throwError s!\"trace witness {witness} is an axiom, not a checked derivation\"\n"
        "    let stx \u2190 match Lean.Parser.runParserCategory env `term expectedSrc with\n"
        "      | .ok s => pure s\n"
        "      | .error e => throwError s!\"cannot parse expected type: {e}\"\n"
        "    let expected \u2190 liftTermElabM do\n"
        "      let e \u2190 Lean.Elab.Term.elabType stx\n"
        "      Lean.Elab.Term.synthesizeSyntheticMVarsNoPostponing\n"
        "      instantiateMVars e\n"
        "    unless \u2190 liftTermElabM (Meta.isDefEq expected info.type) do\n"
        "      throwError s!\"trace witness {witness} has the wrong type: {info.type}\"\n"
        "    let axioms \u2190 Lean.collectAxioms witness.toName\n"
        "    if !axioms.isEmpty then\n"
        "      throwError s!\"trace witness {witness} depends on a nonempty axiom set: {axioms.toList}\"\n"
        "    logInfo s!\"TRACE_OK {witness}\"\n"
    )


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


def _lean_argv(timeout: int) -> list[str]:
    if shutil.which("systemd-run") is None:
        raise ToolError("systemd-run is required for the cgroup lane; it is not on PATH")
    return [
        "systemd-run",
        "--user",
        "--scope",
        "-p",
        "MemoryMax=6G",
        "-p",
        "MemorySwapMax=0",
        "--",
        "timeout",
        str(timeout),
        "env",
        "LEAN_NUM_THREADS=1",
    ]


def _run_lean(argv: list[str], timeout: int) -> tuple[int, str]:
    full = _lean_argv(timeout) + argv
    environment = dict(os.environ)
    environment["LEAN_NUM_THREADS"] = "1"
    try:
        completed = subprocess.run(
            full,
            cwd=str(CORPUS),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout + 120,
        )
    except subprocess.TimeoutExpired as exc:
        raise InconclusiveError(f"timeout after {timeout}s: {' '.join(argv)}") from exc
    except OSError as exc:
        raise ToolError(f"subprocess failed to start ({' '.join(argv)}): {exc}") from exc
    return completed.returncode, completed.stdout or ""


def module_source_path(import_name: str) -> Path:
    return CORPUS / (import_name.replace(".", "/") + ".lean")


def module_olean_path(import_name: str) -> Path:
    return CORPUS / ".lake/build/lib/lean" / (import_name.replace(".", "/") + ".olean")


def build_generated_module(import_name: str, timeout: int) -> tuple[int, str]:
    """Compile one generated module onto the shared `proof/corpus/.lake` lane.

    The generated trace modules are not Lake roots (the lakefile is outside the
    producer task's ownership), so `lake build <module>` reports
    `unknown target`. The module is compiled directly with
    `lake env lean -o <olean>` so its `.olean` lands on `LEAN_PATH`; the probe's
    `import <module>` then resolves. A missing source is a build failure, never a
    silent pass.
    """
    source = module_source_path(import_name)
    if not source.is_file():
        return 1, f"generated module source missing: {source}"
    olean = module_olean_path(import_name)
    olean.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative = source.relative_to(CORPUS)
    except ValueError:
        relative = source
    return _run_lean(["lake", "env", "lean", "-o", str(olean), str(relative)], timeout)


def real_runner(probe_text: str, import_name: str, timeout: int) -> tuple[int, str]:
    """Write the probe under the system temp dir and run it on the shared lane."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(probe_text.encode("utf-8")).hexdigest()[:16]
    probe = WORK_DIR / f"probe-{digest}.lean"
    probe.write_text(probe_text, encoding="utf-8")
    try:
        with lane_lock():
            build_rc, build_out = build_generated_module(import_name, timeout)
            if build_rc != 0:
                return build_rc, "BUILD\n" + build_out
            return _run_lean(["lake", "env", "lean", str(probe)], timeout)
    finally:
        probe.unlink(missing_ok=True)


def _tail(text: str, limit: int = 500) -> str:
    return (text or "").strip()[-limit:]


def generated_module(report: Report) -> str:
    """Derive the Lean import name of the first recorded generated module."""
    generated = [artifact for artifact in report.artifacts if artifact.kind == "generated"]
    if not generated:
        raise ReportError("report records no generated module")
    path = generated[0].path
    if not path.endswith(".lean"):
        raise ReportError(f"generated module is not a .lean file: {path}")
    relative = path[len("proof/corpus/") :] if path.startswith("proof/corpus/") else path
    module = relative[:-5].replace("/", ".")
    if LEAN_MODULE.fullmatch(module) is None:
        raise ReportError(f"generated module name is not a Lean module: {module}")
    return module


def check_kernel(
    report: Report,
    spec: PhaseSpec,
    runner=real_runner,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[str]:
    """Kernel-check every recorded witness through the generated probe."""
    witnesses = tuple(node for node in report.nodes if node.witness)
    if not witnesses:
        raise ReportError(f"phase {spec.id}: no generated witness to kernel-check")
    module = generated_module(report)
    probe = kernel_probe_text(module, witnesses)
    returncode, output = runner(probe, module, timeout)
    lowered = output.lower()
    if returncode in OOM_CODES or "out of memory" in lowered or "oom" in lowered or "killed" in lowered:
        raise InconclusiveError(
            f"phase {spec.id}: kernel check inconclusive (oom/kill, exit {returncode}): {_tail(output)}"
        )
    if returncode != 0:
        raise KernelError(
            f"phase {spec.id}: kernel check rejected the generated witnesses "
            f"(exit {returncode}): {_tail(output)}"
        )
    reported = set(OK_LINE.findall(output))
    missing = [node.witness for node in witnesses if node.witness not in reported]
    if missing:
        raise FalseGreenError(
            f"phase {spec.id}: kernel exited 0 without confirming witness(es): {missing}"
        )
    return sorted(reported)


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------

def load_report(path: Path) -> Report:
    if not path.is_file():
        raise ProducerAbsentError(
            f"producer report absent: {path}; the untrusted generator "
            "(tools/slang/export.py) has not emitted this phase"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportError(f"producer report is not valid JSON: {exc}") from exc
    return parse_report(data)


def check_report(
    report: Report,
    spec: PhaseSpec,
    repo_root: Path,
    song_root: Path,
    runner=real_runner,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[str]:
    if report.phase != spec.id:
        raise ReportError(f"report phase {report.phase!r} != requested {spec.id!r}")
    check_stale(report, repo_root, song_root)
    check_nodes(report, spec)
    check_operations(report, spec)
    check_witness_types(report, spec)
    return check_kernel(report, spec, runner, timeout)


def check_phase(
    phase_id: str,
    repo_root: Path = REPO,
    song_root: Path = DEFAULT_SONG_ROOT,
    runner=real_runner,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[PhaseSpec, Report, list[str]]:
    spec = select_phase(load_phases(), phase_id)
    report = load_report(report_path_for(phase_id))
    witnesses = check_report(report, spec, repo_root, song_root, runner, timeout)
    return spec, report, witnesses


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def run_real_controls(
    phase_id: str,
    repo_root: Path = REPO,
    song_root: Path = DEFAULT_SONG_ROOT,
    runner=real_runner,
    timeout: int = DEFAULT_TIMEOUT,
) -> int:
    """Baseline green, a corrupt derivation red, restored green for one phase."""
    spec, report, witnesses = check_phase(phase_id, repo_root, song_root, runner, timeout)
    print(f"CONTROL {phase_id} baseline: GREEN ({len(witnesses)} witness(es))")

    lemmas = [node for node in report.nodes if node.kind == "lemma"]
    victim = lemmas[0]
    corrupt_witness = "Slang.Generated.Trace.__corrupt_derivation__"
    corrupted = dataclasses.replace(
        report,
        nodes=tuple(
            dataclasses.replace(node, witness=corrupt_witness) if node is victim else node
            for node in report.nodes
        ),
    )
    try:
        check_kernel(corrupted, spec, runner, timeout)
    except (KernelError, FalseGreenError) as exc:
        print(f"CONTROL {phase_id} corrupt-derivation: RED :: {exc}")
    else:
        raise ControlError(f"phase {phase_id}: corrupt derivation survived the kernel check")

    check_kernel(report, spec, runner, timeout)
    print(f"CONTROL {phase_id} restored-green: GREEN")
    return EXIT_OK


def run_controls(repo_root: Path, song_root: Path) -> int:
    """Synthetic controls plus the real-phase baseline; fail closed if blocked."""
    import trace_selftest

    synthetic = trace_selftest.synthetic_controls()
    blocked: list[tuple[str, str]] = []
    for phase_id in PHASE_ORDER:
        if not report_path_for(phase_id).is_file():
            blocked.append(
                (
                    phase_id,
                    f"producer report absent: {report_path_for(phase_id)}; "
                    "tools/slang/export.py has not emitted this phase",
                )
            )
            continue
        try:
            run_real_controls(phase_id, repo_root, song_root)
        except GateError as exc:
            blocked.append((phase_id, f"{type(exc).__name__}: {exc}"))
    if synthetic != EXIT_OK:
        print("FAIL: synthetic controls did not pass", file=sys.stderr)
        return EXIT_REJECT
    if blocked:
        sys.stdout.flush()
        for phase_id, reason in blocked:
            print(f"BLOCKED {phase_id}: {reason}", file=sys.stderr)
        print(
            "FAIL: real baseline/red/restored controls cannot run without the producer report; "
            "a control that cannot run is not a pass",
            file=sys.stderr,
        )
        return EXIT_REJECT
    print("CONTROLS PASS: baseline green, corrupt derivation red, restored green")
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_list(repo_root: Path, song_root: Path) -> int:
    try:
        phases = load_phases()
    except RegistryError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_REJECT
    for phase_id in PHASE_ORDER:
        spec = phases[phase_id]
        report = report_path_for(phase_id)
        status = "report-present" if report.is_file() else "awaiting-producer"
        print(f"{phase_id:<12} [{status}] module={spec.module}")
        print(f"    source: {spec.path}")
        print(f"    report: {report}")
        print(f"    requests: {', '.join(spec.requests) if spec.requests else '-'}")
        for declaration in spec.declarations:
            print(f"    lemma: {declaration}")
    return EXIT_OK


def cmd_phase(args: argparse.Namespace) -> int:
    try:
        phases = load_phases()
        spec = select_phase(phases, args.phase)
    except GateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_REJECT
    song_root = Path(args.song_root).resolve()
    try:
        if args.controls:
            return run_real_controls(spec.id, REPO, song_root)
        _, _, witnesses = check_phase(spec.id, REPO, song_root)
    except ToolError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_TOOL
    except GateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_REJECT
    print(f"PHASE {spec.id} GREEN: {len(witnesses)} witness(es) kernel-checked")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trace_check.py",
        description="Rejecting gate for the W24 generated DecodeTrace.Run derivations.",
    )
    parser.add_argument("--list", action="store_true", help="list the registered phases")
    parser.add_argument("--phase", help="registered phase to check")
    parser.add_argument("--controls", action="store_true", help="run the control suite")
    parser.add_argument("--self-test", action="store_true", help="verify the checker logic")
    parser.add_argument(
        "--song-root",
        default=str(DEFAULT_SONG_ROOT),
        help=f"Song checkout root (default: {DEFAULT_SONG_ROOT})",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        import trace_selftest

        return trace_selftest.self_test()
    if args.list:
        return cmd_list(REPO, Path(args.song_root).resolve())
    if args.controls and not args.phase:
        return run_controls(REPO, Path(args.song_root).resolve())
    if not args.phase:
        parser.error("one of --list, --self-test, --controls or --phase is required")
    return cmd_phase(args)


if __name__ == "__main__":
    sys.exit(main())
