#!/usr/bin/env python3
"""Self-test and synthetic controls for the W24 derivation gate.

Extracted from ``trace_check.py`` to keep each verified-checking file within the
1500-line bound. It imports the checker by name and exercises every rejection
path on synthetic in-memory fixtures; no generated module and no Lean run is
required.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from trace_check import *  # noqa: F401,F403

PHASE = PhaseSpec(
    id="synthetic",
    title="synthetic trace phase",
    module="Slang.Synthetic.Trace",
    path="proof/corpus/Slang/Synthetic/Trace.lean",
    requests=("alpha", "beta"),
    declarations=(
        "Slang.Synthetic.Trace.alpha_run",
        "Slang.Synthetic.Trace.beta_run",
    ),
)

MODULE_REL = "proof/corpus/Slang/Synthetic/Trace.lean"
BYTES_REL = "song/table.slang"
PRISTINE_MODULE = (
    "import Slang.Derivation.Decode\n"
    "namespace Slang.Synthetic.Trace\n"
    "end Slang.Synthetic.Trace\n"
)
MUTANT_MODULE = PRISTINE_MODULE.replace("namespace", "MUTANT namespace", 1)
WITNESS = {
    name: "Slang.Synthetic.Witness." + name.rsplit(".", 1)[-1] for name in PHASE.declarations
}


def _write_repo(root: Path) -> Path:
    module = root / MODULE_REL
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(PRISTINE_MODULE, encoding="utf-8")
    source = root / BYTES_REL
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("let TABLE_PHI = 0;\n", encoding="utf-8")
    return root


def _report_dict(
    root: Path,
    *,
    phase: str = PHASE.id,
    operations: tuple[str, ...] = ("ask", "done"),
    drop_request: str | None = None,
    drop_lemma: str | None = None,
    extra_lemma: dict | None = None,
    type_override: str | None = None,
    request_type_override: str | None = None,
    module_digest: str | None = None,
    bytes_digest: str | None = None,
    schema: int = 1,
    revision: str = "a" * 40,
    artifact_override: list[dict] | None = None,
) -> dict:
    module = root / MODULE_REL
    source = root / BYTES_REL
    nodes: list[dict] = []
    for name in PHASE.requests:
        if name == drop_request:
            continue
        request_type = request_type_override if request_type_override is not None else (
            "Slang.Derivation.DecodeTrace.Run Slang.Synthetic.Trace." + name + "_run"
        )
        nodes.append(
            {
                "kind": "request",
                "name": name,
                "witness": "Slang.Synthetic.Witness." + name + "_call",
                "type": request_type,
            }
        )
    for declaration in PHASE.declarations:
        if declaration == drop_lemma:
            continue
        expected = type_override if type_override is not None else (
            "Slang.Derivation.DecodeTrace.Run " + declaration
        )
        nodes.append(
            {
                "kind": "lemma",
                "name": declaration,
                "witness": WITNESS[declaration],
                "type": expected,
            }
        )
    if extra_lemma is not None:
        nodes.append(extra_lemma)
    artifacts = artifact_override or [
        {"path": MODULE_REL, "sha256": module_digest or sha256_file(module), "kind": "generated"},
        {"path": BYTES_REL, "sha256": bytes_digest or sha256_file(source), "kind": "bytes"},
    ]
    return {
        "schema": schema,
        "phase": phase,
        "song_revision": revision,
        "artifacts": artifacts,
        "nodes": nodes,
        "operations": list(operations),
    }


class FakeRunner:
    """A deterministic stand-in for the Lean lane; reads the module on disk."""

    def __init__(self, module: Path, confirm: tuple[str, ...]):
        self.module = module
        self.confirm = confirm

    def __call__(self, probe_text: str, import_name: str, timeout: int) -> tuple[int, str]:
        if "MUTANT" in self.module.read_text(encoding="utf-8"):
            return 1, "error: mutated derivation rejected"
        return 0, "\n".join(f"TRACE_OK {name}" for name in self.confirm)


class FailRunner:
    def __call__(self, probe_text: str, import_name: str, timeout: int) -> tuple[int, str]:
        return 1, "error: trace witness has the wrong type"


class OomRunner:
    def __call__(self, probe_text: str, import_name: str, timeout: int) -> tuple[int, str]:
        return 137, "Killed (out of memory)"


class TimeoutRunner:
    def __call__(self, probe_text: str, import_name: str, timeout: int) -> tuple[int, str]:
        raise InconclusiveError("timeout after 1s: lake env lean probe.lean")


def _run_checks(root: Path) -> list[dict]:
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    def parse(**kwargs) -> Report:
        return parse_report(_report_dict(root, **kwargs))

    def expect_reject(name: str, fn, exc_type, *needles: str) -> None:
        try:
            fn()
        except exc_type as exc:
            ok = all(needle in str(exc) for needle in needles)
            record(name, ok, f"raised {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - the self-test reports the surprise
            record(name, False, f"wrong exception {type(exc).__name__}: {exc}")
        else:
            record(name, False, "rejection path did not reject")

    module = root / MODULE_REL
    baseline_report = parse()
    confirm = tuple(sorted({node.witness for node in baseline_report.nodes if node.witness}))
    green = FakeRunner(module, confirm)

    # 0. the positive: fresh digests, every node present, kernel green.
    report = parse()
    try:
        witnesses = check_report(report, PHASE, root, root / "song", green)
        record(
            "positive_accept",
            witnesses == sorted(confirm),
            f"witnesses={witnesses}",
        )
    except GateError as exc:
        record("positive_accept", False, f"{type(exc).__name__}: {exc}")

    # 1. stale bytes: digest mismatch and a missing generated file.
    expect_reject(
        "reject_stale_digest",
        lambda: check_stale(parse(module_digest="0" * 64), root, root / "song"),
        StaleError,
        "stale generated bytes",
    )
    absent = [{"path": "proof/corpus/Slang/Synthetic/Absent.lean", "sha256": "0" * 64, "kind": "generated"}]
    expect_reject(
        "reject_missing_artifact",
        lambda: check_stale(parse(artifact_override=absent), root, root / "song"),
        StaleError,
        "artifact missing",
    )
    escape = [{"path": "../../etc/passwd", "sha256": "0" * 64, "kind": "generated"}]
    expect_reject(
        "reject_path_escape",
        lambda: check_stale(parse(artifact_override=escape), root, root / "song"),
        StaleError,
        "escapes its root",
    )

    # 2. missing node: an expected request constructor or lemma absent.
    expect_reject(
        "reject_missing_request_node",
        lambda: check_nodes(parse(drop_request="beta"), PHASE),
        MissingNodeError,
        "missing request node",
    )
    expect_reject(
        "reject_missing_lemma_node",
        lambda: check_nodes(parse(drop_lemma=PHASE.declarations[0]), PHASE),
        MissingNodeError,
        "missing lemma node",
    )

    # 3. unsupported producer operation and unknown request constructor.
    expect_reject(
        "reject_unsupported_operation",
        lambda: check_operations(parse(operations=("ask", "bogus")), PHASE),
        UnsupportedOperationError,
        "unsupported producer operation",
    )
    unknown = {
        "kind": "request",
        "name": "gamma",
        "witness": "Slang.Synthetic.Witness.gamma_run",
        "type": "Slang.Derivation.DecodeTrace.Run Slang.Synthetic.Trace.gamma_run",
    }
    extra_nodes = _report_dict(root)
    extra_nodes["nodes"].append(unknown)
    expect_reject(
        "reject_unknown_request_constructor",
        lambda: check_operations(parse_report(extra_nodes), PHASE),
        UnsupportedOperationError,
        "unknown request constructor",
    )

    # 4. false green and kernel rejection.
    bogus = {
        "kind": "lemma",
        "name": "Slang.Synthetic.Trace.bogus_run",
        "witness": "Slang.Synthetic.Witness.bogus_run",
        "type": "Slang.Derivation.DecodeTrace.Run Slang.Synthetic.Trace.bogus_run",
    }
    expect_reject(
        "reject_false_green_kernel_target",
        lambda: check_kernel(parse(extra_lemma=bogus), PHASE, green),
        FalseGreenError,
        "without confirming witness",
    )
    expect_reject(
        "reject_kernel_nonzero",
        lambda: check_kernel(parse(), PHASE, FailRunner()),
        KernelError,
        "rejected the generated witnesses",
    )
    expect_reject(
        "reject_timeout_inconclusive",
        lambda: check_kernel(parse(), PHASE, TimeoutRunner()),
        InconclusiveError,
        "timeout",
    )
    expect_reject(
        "reject_oom_inconclusive",
        lambda: check_kernel(parse(), PHASE, OomRunner()),
        InconclusiveError,
        "inconclusive",
    )

    # 5. malformed reports and non-Run lemma types.
    expect_reject(
        "reject_bad_schema",
        lambda: parse(schema=2),
        ReportError,
        "schema must be 1",
    )
    expect_reject(
        "reject_bad_revision",
        lambda: parse(revision="nothex"),
        ReportError,
        "must be 40 hex",
    )
    no_generated = [{"path": BYTES_REL, "sha256": "0" * 64, "kind": "bytes"}]
    expect_reject(
        "reject_no_generated_module",
        lambda: parse(artifact_override=no_generated),
        ReportError,
        "at least one generated module",
    )
    duplicate = _report_dict(root)
    duplicate["nodes"].append(dict(duplicate["nodes"][0]))
    expect_reject(
        "reject_duplicate_node",
        lambda: parse_report(duplicate),
        ReportError,
        "duplicate node",
    )
    witnessless = _report_dict(root)
    witnessless["nodes"][-1].pop("witness")
    expect_reject(
        "reject_lemma_without_witness",
        lambda: parse_report(witnessless),
        ReportError,
        "witness must be a nonempty string",
    )
    expect_reject(
        "reject_non_run_type",
        lambda: check_witness_types(parse(request_type_override="True"), PHASE),
        WitnessTypeError,
        "non-Run type",
    )
    # A structural lemma (`evaluate`, `body_eq`) is not required to be a Run.
    try:
        check_witness_types(parse(type_override="True"), PHASE)
        record("accept_structural_lemma_type", True, "non-Run lemma type accepted")
    except GateError as exc:
        record("accept_structural_lemma_type", False, f"{type(exc).__name__}: {exc}")

    # 6. a report for another phase and a registry without the module.
    expect_reject(
        "reject_wrong_phase",
        lambda: check_report(parse(phase="other"), PHASE, root, root / "song", green),
        ReportError,
        "!= requested",
    )
    try:
        select_phase({"synthetic": PHASE}, "absent")
    except PhaseError as exc:
        record("reject_missing_phase", True, f"raised PhaseError: {exc}")
    else:
        record("reject_missing_phase", False, "missing phase survived")

    return checks


def self_test() -> int:
    """Exercise every rejection path; exit 0 only if each rejects and good accepts."""
    base = Path(tempfile.gettempdir()) / "opencode"
    base.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="trace-selftest-", dir=str(base)))
    try:
        root = _write_repo(workdir / "repo")
        checks = _run_checks(root)
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
    return EXIT_OK if not failed else EXIT_REJECT


def synthetic_controls() -> int:
    """Baseline green, generated-kernel mutation red, restored green, plus rejections.

    No Lean run is required: a deterministic runner stands in for the lane. This
    demonstrates the control logic; the real baseline still fails closed while the
    producer report is absent.
    """
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"control": name, "ok": ok, "detail": detail})
        marker = "GREEN" if ok else "FAIL"
        print(f"CONTROL {name}: {marker} :: {detail}")

    def expect_red(name: str, fn, exc_type) -> None:
        try:
            fn()
        except exc_type as exc:
            record(name, True, f"RED {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - the control reports the surprise
            record(name, False, f"wrong exception {type(exc).__name__}: {exc}")
        else:
            record(name, False, "control did not reject")

    base = Path(tempfile.gettempdir()) / "opencode"
    base.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="trace-controls-", dir=str(base)))
    try:
        root = _write_repo(workdir / "repo")
        module = root / MODULE_REL
        baseline_report = parse_report(_report_dict(root))
        confirm = tuple(sorted({node.witness for node in baseline_report.nodes if node.witness}))
        green = FakeRunner(module, confirm)

        # baseline: fresh digests, all nodes, kernel green.
        baseline = parse_report(_report_dict(root))
        try:
            check_report(baseline, PHASE, root, root / "song", green)
            record("baseline-green", True, "fresh bytes and kernel witnesses green")
        except GateError as exc:
            record("baseline-green", False, f"{type(exc).__name__}: {exc}")

        # mutation: corrupt the generated kernel and refresh the digest so the
        # stale check passes; the kernel must reject the derivation.
        module.write_text(MUTANT_MODULE, encoding="utf-8")
        mutated = parse_report(_report_dict(root))
        try:
            check_report(mutated, PHASE, root, root / "song", green)
        except KernelError as exc:
            record("mutation-red", True, f"KernelError: {exc}")
        except GateError as exc:
            record("mutation-red", False, f"unexpected {type(exc).__name__}: {exc}")
        else:
            record("mutation-red", False, "mutated derivation survived the kernel check")

        # restored: pristine module and digest again.
        module.write_text(PRISTINE_MODULE, encoding="utf-8")
        restored = parse_report(_report_dict(root))
        try:
            check_report(restored, PHASE, root, root / "song", green)
            record("restored-green", True, "restored bytes and witnesses green")
        except GateError as exc:
            record("restored-green", False, f"{type(exc).__name__}: {exc}")

        # stale bytes: correct content, stale recorded digest.
        stale = parse_report(_report_dict(root, module_digest="0" * 64))
        expect_red(
            "reject-stale-bytes",
            lambda: check_stale(stale, root, root / "song"),
            StaleError,
        )

        # missing node: an expected request constructor absent.
        missing = parse_report(_report_dict(root, drop_request="alpha"))
        expect_red(
            "reject-missing-node",
            lambda: check_nodes(missing, PHASE),
            MissingNodeError,
        )

        # unsupported operation.
        unsupported = parse_report(_report_dict(root, operations=("ask", "bogus")))
        expect_red(
            "reject-unsupported-operation",
            lambda: check_operations(unsupported, PHASE),
            UnsupportedOperationError,
        )

        # false-green kernel target: a declared witness the kernel never confirms.
        bogus = {
            "kind": "lemma",
            "name": "Slang.Synthetic.Trace.bogus_run",
            "witness": "Slang.Synthetic.Witness.bogus_run",
            "type": "Slang.Derivation.DecodeTrace.Run Slang.Synthetic.Trace.bogus_run",
        }
        false_green = parse_report(_report_dict(root, extra_lemma=bogus))
        expect_red(
            "reject-false-green-kernel-target",
            lambda: check_kernel(false_green, PHASE, green),
            FalseGreenError,
        )

        # a timeout or an OOM is inconclusive, never a green control.
        current = parse_report(_report_dict(root))
        expect_red(
            "reject-timeout-inconclusive",
            lambda: check_kernel(current, PHASE, TimeoutRunner()),
            InconclusiveError,
        )
        expect_red(
            "reject-oom-inconclusive",
            lambda: check_kernel(current, PHASE, OomRunner()),
            InconclusiveError,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    failed = [check for check in checks if not check["ok"]]
    print(
        json.dumps(
            {
                "synthetic_controls": "pass" if not failed else "fail",
                "controls": len(checks),
            },
            sort_keys=True,
        )
    )
    return EXIT_OK if not failed else EXIT_REJECT
