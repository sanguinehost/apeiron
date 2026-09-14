#!/usr/bin/env python3
"""S8 assurance perimeter checker.

Definition of "trusted checking code" used here
===============================================
Trusted checking code is the local, hand-written code that produces or
adjudicates acceptance evidence and whose verdict is trusted by inspection
rather than established by a kernel check or by the Rust test suite. It is the
local trusted computing base (TCB) for this repository:

  * the extraction freshness/regeneration driver (``proof/regen.sh``);
  * the toolchain patch installer (``proof/install-patches.sh``);
  * the storage mutation-control runner (``proof/storage-controls.py``);
  * the axiom-cone audit driver (``proof/corpus/AxiomAudit.lean``);
  * the bounded-model-checking harnesses (``src/proofs.rs``);
  * the extraction selection wrapper (``proof/extract/lib.rs``);
  * the hand-written boundary statement modules (``proof/Denotation.lean``,
    ``proof/Refinement.lean``) and the package entry points
    (``proof/lakefile.lean``, ``proof/corpus/lakefile.lean``);
  * this checker (``tools/check-assurance.py``).

Justification against AGENTS.md section 4: "Local trusted checking code
<= 5,000 physical lines, owned source files <= 1,500 lines, <= 12 immediate
source files per directory. Inventory external trust separately; neither the
whole proof library nor a bridge line count is the full TCB." This checker
bounds the manifest above at 5,000 lines, bounds every first-party owned source
file at 1,500 lines and every owned directory at 12 immediate source files, and
reports external trust separately, excluded from the 5,000-line local total.

The excluded external trust is inventoried, not counted: the pinned Lean kernel
and Mathlib, the canonical Apeiron corpus, the Charon/Aeneas toolchain, the
generated extraction (``proof/ApeironWord.lean``) and the hand-written Lean
proof bridge. Kernel-checked Lean proofs are not trusted checking code -- the
kernel checks them -- so their line counts are reported, never added to the
local TCB. The call-census tooling is a separate measurement tool, not part of
the Apeiron acceptance gates; it is owned source subject to the per-file bound
but not to the 5,000-line local total.

Owned source is every first-party source file under the declared owned roots
(``src``, ``tests``, ``examples``, ``benches``, ``fuzz``, ``derive``,
``fold-abi``, ``tools``, ``proof``, ``docs``); vendored and build directories
(``.git``, ``.lake``, ``target``) are excluded, and the generated extraction is
excluded by name. A source file outside the owned roots is unlisted.

Physical lines are newline-separated lines including blank and comment lines,
not logical statements.

A source file that is not under one of the declared owned roots is an unlisted
production source and is rejected. A declared member that is absent, and an
empty trusted-checking population, are rejected: the checker never passes
silently.

Exit status: 0 when every bound holds and every declared member exists; 1 on
any breach or missing declared member; 2 on a usage or I/O error.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]

LIMIT_TOTAL = 5000
LIMIT_FILE = 1500
LIMIT_DIR = 12

SOURCE_EXTS = {".rs", ".lean", ".py", ".sh"}
GENERATED_NAMES = {"ApeironWord.lean"}
EXCLUDE_DIR_NAMES = {
    ".git",
    ".lake",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    ".mut-s6",
}

# First-party source lives only under these roots. A source file anywhere else
# is an unlisted production source.
OWNED_ROOTS = (
    "src",
    "tests",
    "examples",
    "benches",
    "fuzz",
    "derive",
    "fold-abi",
    "tools",
    "proof",
    "docs",
)

# Local trusted checking code, explicit manifest. Every entry must exist.
TRUSTED_CHECKING = (
    "tools/check-assurance.py",
    "src/proofs.rs",
    "proof/regen.sh",
    "proof/install-patches.sh",
    "proof/storage-controls.py",
    "proof/extract/lib.rs",
    "proof/corpus/AxiomAudit.lean",
    "proof/Denotation.lean",
    "proof/Refinement.lean",
    "proof/lakefile.lean",
    "proof/corpus/lakefile.lean",
)


def physical_lines(path):
    """Count newline-separated physical lines, tolerating non-UTF-8 bytes."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(text.splitlines())


def rel(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def iter_source_files(root):
    """All first-party source files under root, excluding build/vendor dirs."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
        for name in filenames:
            if name in GENERATED_NAMES:
                continue
            path = Path(dirpath) / name
            if path.suffix in SOURCE_EXTS:
                found.append(path)
    return found


def immediate_source_count(directory):
    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0
    return sum(
        1
        for entry in entries
        if entry.is_file() and entry.suffix in SOURCE_EXTS and entry.name not in GENERATED_NAMES
    )


def tree_totals(directory, extensions):
    """(file_count, byte_count) for files with the given extensions."""
    files = 0
    size = 0
    if not directory.is_dir():
        return files, size
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".lake", "target"}]
        for name in filenames:
            if Path(name).suffix in extensions:
                files += 1
                try:
                    size += (Path(dirpath) / name).stat().st_size
                except OSError:
                    pass
    return files, size


def tree_lines(directory, extensions):
    files = 0
    lines = 0
    if not directory.is_dir():
        return files, lines
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".lake", "target"}]
        for name in filenames:
            if Path(name).suffix in extensions:
                try:
                    lines += physical_lines(Path(dirpath) / name)
                    files += 1
                except OSError:
                    pass
    return files, lines


def build_external_inventory(root):
    """External trust, reported separately and never counted in the TCB."""
    entries = []

    corpus = root / "proof" / "corpus" / ".lake" / "packages" / "apeiron" / "proof" / "Apeiron"
    files, lines = tree_lines(corpus, {".lean"})
    entries.append(
        {
            "name": "canonical-corpus",
            "path": rel(corpus, root),
            "kind": "pinned Lean corpus (kernel-checked)",
            "counted_in_tcb": False,
            "present": corpus.is_dir(),
            "files": files,
            "lines": lines,
        }
    )

    for label, packages in (
        ("mathlib-and-dependencies", root / "proof" / "corpus" / ".lake" / "packages"),
        ("proof-package-dependencies", root / "proof" / ".lake" / "packages"),
    ):
        files, size = tree_totals(packages, {".lean", ".toml", ".json", ".sh"})
        entries.append(
            {
                "name": label,
                "path": rel(packages, root),
                "kind": "pinned Lean dependency tree (kernel/toolchain trust)",
                "counted_in_tcb": False,
                "present": packages.is_dir(),
                "files": files,
                "bytes": size,
                "lines": None,
                "lines_note": "not measured: line-counting the dependency tree is not part of the local gate",
            }
        )

    bridge_files = []
    generated = []
    proof_root = root / "proof"
    if proof_root.is_dir():
        for dirpath, dirnames, filenames in os.walk(proof_root):
            dirnames[:] = [d for d in dirnames if d not in {".git", ".lake", "target"}]
            for name in filenames:
                if Path(name).suffix != ".lean":
                    continue
                path = Path(dirpath) / name
                if name in GENERATED_NAMES:
                    generated.append(path)
                else:
                    bridge_files.append(path)
    bridge_lines = sum(physical_lines(p) for p in bridge_files)
    generated_lines = sum(physical_lines(p) for p in generated)
    entries.append(
        {
            "name": "local-proof-bridge",
            "path": "proof",
            "kind": "hand-written Lean bridge (not TCB; line count not the local bound)",
            "counted_in_tcb": False,
            "files": len(bridge_files),
            "lines": bridge_lines,
        }
    )
    entries.append(
        {
            "name": "generated-extraction",
            "path": "proof/ApeironWord.lean",
            "kind": "generated Aeneas output (not hand-written, not TCB)",
            "counted_in_tcb": False,
            "files": len(generated),
            "lines": generated_lines,
        }
    )

    toolchains = {}
    for relpath in ("proof/lean-toolchain", "proof/corpus/lean-toolchain"):
        path = root / relpath
        toolchains[relpath] = path.read_text(encoding="utf-8").strip() if path.is_file() else None
    entries.append(
        {
            "name": "lean-toolchain",
            "path": "proof/lean-toolchain, proof/corpus/lean-toolchain",
            "kind": "pinned Lean toolchain (trusted)",
            "counted_in_tcb": False,
            "revisions": toolchains,
        }
    )

    aeneas = root / "target" / "aeneas"
    binaries = {}
    for name in ("charon", "aeneas"):
        path = aeneas / name
        binaries[name] = path.stat().st_size if path.is_file() else None
    entries.append(
        {
            "name": "charon-aeneas",
            "path": "target/aeneas",
            "kind": "pinned translation toolchain (external trust)",
            "counted_in_tcb": False,
            "present": aeneas.is_dir(),
            "binary_bytes": binaries,
        }
    )
    return entries


def build_report(root):
    root = Path(root).resolve()
    failures = []

    # --- trusted checking code -------------------------------------------
    trusted_entries = []
    for relpath in TRUSTED_CHECKING:
        path = root / relpath
        if not path.is_file():
            failures.append(f"trusted-checking: declared member missing: {relpath}")
            continue
        trusted_entries.append({"path": relpath, "lines": physical_lines(path)})
    if not trusted_entries:
        failures.append("trusted-checking: population is empty")

    trusted_total = sum(e["lines"] for e in trusted_entries)
    trusted_file_over = [e for e in trusted_entries if e["lines"] > LIMIT_FILE]
    if trusted_total > LIMIT_TOTAL:
        failures.append(f"trusted-checking: total {trusted_total} > {LIMIT_TOTAL}")
    for entry in trusted_file_over:
        failures.append(
            f"trusted-checking: {entry['path']} has {entry['lines']} lines > {LIMIT_FILE}"
        )
    trusted_dirs = sorted({(root / e["path"]).parent for e in trusted_entries})
    trusted_dir_over = []
    for directory in trusted_dirs:
        count = immediate_source_count(directory)
        if count > LIMIT_DIR:
            trusted_dir_over.append({"directory": rel(directory, root), "count": count})
            failures.append(
                f"trusted-checking: {rel(directory, root)} has {count} immediate source files > {LIMIT_DIR}"
            )

    # --- owned source -----------------------------------------------------
    discovered = iter_source_files(root)
    owned = []
    unlisted = []
    for path in discovered:
        top = rel(path, root).split("/", 1)[0]
        if top in OWNED_ROOTS:
            owned.append(path)
        else:
            unlisted.append(path)

    owned_entries = []
    for path in sorted(owned):
        owned_entries.append({"path": rel(path, root), "lines": physical_lines(path)})

    owned_file_over = [e for e in owned_entries if e["lines"] > LIMIT_FILE]
    for entry in owned_file_over:
        failures.append(f"owned-source: {entry['path']} has {entry['lines']} lines > {LIMIT_FILE}")

    owned_dirs = {}
    for path in owned:
        owned_dirs.setdefault(path.parent, 0)
    owned_dir_over = []
    for directory in sorted(owned_dirs):
        count = immediate_source_count(directory)
        owned_dirs[directory] = count
        if count > LIMIT_DIR:
            owned_dir_over.append({"directory": rel(directory, root), "count": count})
            failures.append(
                f"owned-source: {rel(directory, root)} has {count} immediate source files > {LIMIT_DIR}"
            )

    for unlisted_path in sorted(unlisted):
        failures.append(f"owned-source: unlisted production source: {rel(unlisted_path, root)}")

    for owned_root in OWNED_ROOTS:
        if not (root / owned_root).is_dir():
            failures.append(f"owned-source: declared root missing: {owned_root}/")

    owned_total = sum(e["lines"] for e in owned_entries)

    report = {
        "root": str(root),
        "limits": {
            "trusted_total_lines": LIMIT_TOTAL,
            "file_lines": LIMIT_FILE,
            "dir_files": LIMIT_DIR,
            "source_extensions": sorted(SOURCE_EXTS),
        },
        "trusted_checking": {
            "manifest": list(TRUSTED_CHECKING),
            "files": trusted_entries,
            "file_count": len(trusted_entries),
            "total_lines": trusted_total,
            "per_file_over": trusted_file_over,
            "per_dir_over": trusted_dir_over,
        },
        "owned_source": {
            "roots": list(OWNED_ROOTS),
            "file_count": len(owned_entries),
            "total_lines": owned_total,
            "per_file_over": owned_file_over,
            "per_dir_over": owned_dir_over,
            "unlisted": [rel(p, root) for p in sorted(unlisted)],
        },
        "external_trust": build_external_inventory(root),
        "failures": failures,
        "ok": not failures,
    }
    return report


def print_report(report):
    limits = report["limits"]
    print(f"repo: {report['root']}")
    print(
        "limits: trusted-checking <= {} lines total; owned source <= {} lines/file; "
        "<= {} immediate source files/dir".format(
            limits["trusted_total_lines"], limits["file_lines"], limits["dir_files"]
        )
    )

    trusted = report["trusted_checking"]
    print(
        f"[bounded] trusted-checking: {trusted['file_count']} files, "
        f"{trusted['total_lines']} lines"
    )
    for entry in trusted["per_file_over"]:
        print(f"  OVER file: {entry['path']} {entry['lines']} > {limits['file_lines']}")
    for entry in trusted["per_dir_over"]:
        print(
            f"  OVER dir: {entry['directory']} {entry['count']} > {limits['dir_files']}"
        )
    if trusted["total_lines"] <= limits["trusted_total_lines"]:
        print(f"  total {trusted['total_lines']} <= {limits['trusted_total_lines']}")

    owned = report["owned_source"]
    print(
        f"[bounded] owned-source: {owned['file_count']} files, {owned['total_lines']} lines"
    )
    for entry in owned["per_file_over"]:
        print(f"  OVER file: {entry['path']} {entry['lines']} > {limits['file_lines']}")
    for entry in owned["per_dir_over"]:
        print(
            f"  OVER dir: {entry['directory']} {entry['count']} > {limits['dir_files']}"
        )
    for path in owned["unlisted"]:
        print(f"  OVER unlisted production source: {path}")

    print("[inventory] external trust (not counted in the 5,000-line local TCB):")
    for entry in report["external_trust"]:
        if "files" in entry:
            detail = f"{entry['files']} files"
            if entry.get("lines") is not None:
                detail += f", {entry['lines']} lines"
            elif "bytes" in entry:
                detail += f", {entry['bytes']} bytes"
        elif "revisions" in entry:
            detail = ", ".join(
                f"{path}={value}" for path, value in entry["revisions"].items()
            )
        elif "binary_bytes" in entry:
            detail = "present" if entry.get("present") else "absent"
            detail += ": " + ", ".join(
                f"{name}={size if size is not None else 'absent'}"
                for name, size in entry["binary_bytes"].items()
            )
        else:
            detail = ""
        print(f"  {entry['name']}: {detail}")

    if report["failures"]:
        print(f"\nFAIL: {len(report['failures'])} assurance breach(es)")
        for failure in report["failures"]:
            print(f"  - {failure}")
    else:
        print("\nOK: every bounded population is within limits and every declared member exists")


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _synthetic_tree(base):
    """An under-limit throwaway tree containing every declared member."""
    for relpath in TRUSTED_CHECKING:
        path = base / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")
    for owned_root in OWNED_ROOTS:
        (base / owned_root).mkdir(parents=True, exist_ok=True)
    (base / "src" / "lib.rs").write_text("pub fn a() {}\n", encoding="utf-8")
    (base / "tests" / "t.rs").write_text("#[test]\nfn t() {}\n", encoding="utf-8")


def _run_checker(root):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--json",
        "--root",
        str(root),
    ]
    return subprocess.run(command, capture_output=True, text=True)


def self_test():
    workdir = Path(tempfile.mkdtemp(prefix="apeiron-assurance-selftest-"))
    results = []

    def record(name, expected, proc):
        try:
            report = json.loads(proc.stdout)
            failures = report.get("failures", [])
        except (ValueError, AttributeError):
            failures = ["<unparseable JSON output>"]
        passed = proc.returncode == expected and (expected == 0 or bool(failures))
        results.append((name, expected, proc.returncode, passed, failures))

    try:
        base = workdir / "base"
        base.mkdir()
        _synthetic_tree(base)

        record("positive: under-limit copy", 0, _run_checker(base))

        def fresh(name):
            target = workdir / name
            shutil.copytree(base, target)
            return target

        def per_file_over(tree):
            (tree / "src" / "big.rs").write_text("x\n" * 1501, encoding="utf-8")

        def per_dir_over(tree):
            for index in range(13):
                (tree / "examples" / f"e{index}.rs").write_text("x\n", encoding="utf-8")

        def total_over(tree):
            for relpath in (
                "proof/regen.sh",
                "proof/install-patches.sh",
                "proof/extract/lib.rs",
                "proof/Denotation.lean",
            ):
                (tree / relpath).write_text("# pad\n" * 1500, encoding="utf-8")

        def unlisted_source(tree):
            (tree / "stray").mkdir()
            (tree / "stray" / "unlisted.rs").write_text("x\n", encoding="utf-8")

        def missing_member(tree):
            (tree / "proof" / "Refinement.lean").unlink()

        for name, mutate in (
            ("negative: owned file > 1500 lines", per_file_over),
            ("negative: directory > 12 source files", per_dir_over),
            ("negative: trusted total > 5000 lines", total_over),
            ("negative: unlisted production source", unlisted_source),
            ("negative: declared member missing", missing_member),
        ):
            slug = "".join(ch if ch.isalnum() else "-" for ch in name)
            tree = fresh(slug)
            mutate(tree)
            record(name, 1, _run_checker(tree))

        for name, expected, observed, passed, failures in results:
            verdict = "PASS" if passed else "FAIL"
            print(f"[self-test] {name}: expected exit {expected}, observed {observed} -> {verdict}")
            for failure in failures[:3]:
                print(f"            {failure}")

        failures = [name for name, _, _, passed, _ in results if not passed]
        print(f"\n[self-test] {len(results) - len(failures)}/{len(results)} controls behaved as required")
        if failures:
            print("[self-test] FAILED: " + ", ".join(failures))
            return 1
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="repository root to check")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="report the perimeter and bounds")
    group.add_argument("--json", action="store_true", help="machine-readable report")
    group.add_argument("--self-test", action="store_true", help="run throwaway-copy controls")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    try:
        report = build_report(args.root)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
