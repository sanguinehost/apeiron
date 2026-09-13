#!/usr/bin/env python3
"""Measure the declared assurance populations against the size budget.

Bounded populations (enforced; nonzero exit on breach):

  * `trusted-checking` -- the local gate/audit/extraction sources that can
    admit a headline: the two claim/assurance checkers, the audit driver,
    the extraction selection, the translation/model boundary and the
    package entry points.
  * `bridge-proof` -- every hand-written source of the owned `proof/`
    packages (generated `proof/ApeironWord.lean` and build products are
    excluded by name, not by hand-picked subset).

Reported inventory (measured and flagged, not enforced):

  * `owned-rust` -- the `src/` and `tests/` production sources. These are
    outside the local trusted-checking perimeter (AGENTS.md section 11).
    Items over the numeric limits are printed `OVER` so that binding them
    is a reviewed perimeter change, not a silent pass.

Limits (AGENTS.md section 4 / CLAIMS.md perimeter): total <= 5000 physical
lines; per file <= 1500; per directory <= 12 immediate source files. Each
file is counted once in the union total.

No external dependencies. Run from the repository root:

    python3 tools/check-assurance.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LIMIT_TOTAL = 5000
LIMIT_FILE = 1500
LIMIT_DIR = 12
SOURCE_EXTS = {".lean", ".sh", ".rs", ".py"}
GENERATED_NAMES = {"ApeironWord.lean"}  # extracted, not trusted checking code
EXCLUDE_DIRS = {".lake", "target", ".git"}

TRUSTED_CHECKING = [
    "tools/check-claims.py",
    "tools/check-assurance.py",
    "proof/regen.sh",
    "proof/corpus/AxiomAudit.lean",
    "proof/extract/lib.rs",
    "proof/Denotation.lean",
    "proof/Refinement.lean",
    "proof/lakefile.lean",
    "proof/corpus/lakefile.lean",
]

BOUNDED = (
    ("trusted-checking", "files", TRUSTED_CHECKING),
    ("bridge-proof", "tree", "proof"),
)

INVENTORY = (
    ("owned-rust", ("src", "tests"), (".rs",)),
)


def physical_lines(path):
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def bounded_files(kind, spec):
    if kind == "files":
        files, missing = [], []
        for rel in spec:
            path = ROOT / rel
            if path.is_file():
                files.append(path)
            else:
                missing.append(rel)
        return files, missing
    root = ROOT / spec
    if not root.is_dir():
        return [], [spec]
    files = [
        f
        for f in root.rglob("*")
        if f.is_file()
        and f.suffix in SOURCE_EXTS
        and not any(part in EXCLUDE_DIRS for part in f.parts)
        and f.name not in GENERATED_NAMES
    ]
    return files, []


def dir_source_files(directory):
    if not directory.is_dir():
        return []
    return [
        f
        for f in directory.iterdir()
        if f.is_file() and f.suffix in SOURCE_EXTS and f.name not in GENERATED_NAMES
    ]


def measure(files):
    total = 0
    max_file = 0
    max_path = None
    for f in files:
        n = physical_lines(f)
        total += n
        if n > max_file:
            max_file, max_path = n, f
    return total, max_file, max_path


def check_population(name, files, failures):
    total, max_file, max_path = measure(files)
    dirs = sorted({f.parent for f in files})
    print(
        f"[bounded] {name}: {len(files)} files, {total} lines, "
        f"max file {max_file} ({max_path.relative_to(ROOT)})"
    )
    if total > LIMIT_TOTAL:
        failures.append(f"{name}: {total} lines > {LIMIT_TOTAL}")
        print(f"  OVER total: {total} > {LIMIT_TOTAL}")
    if max_file > LIMIT_FILE:
        failures.append(f"{name}: {max_path.relative_to(ROOT)} has {max_file} lines > {LIMIT_FILE}")
        print(f"  OVER file: {max_path.relative_to(ROOT)} {max_file} > {LIMIT_FILE}")
    for directory in dirs:
        count = len(dir_source_files(directory))
        if count > LIMIT_DIR:
            failures.append(
                f"{name}: {directory.relative_to(ROOT)} has {count} immediate source files > {LIMIT_DIR}"
            )
            print(f"  OVER dir: {directory.relative_to(ROOT)} {count} > {LIMIT_DIR}")
    return set(files)


def main():
    failures = []
    print(f"repo: {ROOT}")
    print(
        f"limits: total<={LIMIT_TOTAL} lines, per file<={LIMIT_FILE} lines, "
        f"per directory<={LIMIT_DIR} immediate source files"
    )

    union = set()
    for name, kind, spec in BOUNDED:
        files, missing = bounded_files(kind, spec)
        if missing:
            print(f"MISSING {name}: declared members absent: {', '.join(missing)}")
            failures.append(f"{name}: MISSING declared members")
            continue
        if not files:
            print(f"MISSING {name}: population is defined but empty")
            failures.append(f"{name}: MISSING population")
            continue
        union |= check_population(name, files, failures)

    union_total = sum(physical_lines(f) for f in union)
    print(f"[bounded] perimeter-union: {len(union)} files, {union_total} lines")
    if union_total > LIMIT_TOTAL:
        failures.append(f"perimeter-union: {union_total} lines > {LIMIT_TOTAL}")
        print(f"  OVER union total: {union_total} > {LIMIT_TOTAL}")

    for name, roots, exts in INVENTORY:
        files = []
        missing_roots = []
        for root in roots:
            base = ROOT / root
            if not base.is_dir():
                missing_roots.append(root)
                continue
            files.extend(
                sorted(f for f in base.iterdir() if f.is_file() and f.suffix in exts)
            )
        if missing_roots:
            print(f"MISSING {name}: roots absent: {', '.join(missing_roots)}")
            failures.append(f"{name}: MISSING roots")
            continue
        if not files:
            print(f"MISSING {name}: no source files")
            failures.append(f"{name}: MISSING population")
            continue
        total, max_file, max_path = measure(files)
        print(
            f"[inventory] {name}: {len(files)} files, {total} lines, "
            f"max file {max_file} ({max_path.relative_to(ROOT)})"
        )
        over = []
        if total > LIMIT_TOTAL:
            over.append(f"total={total}")
        if max_file > LIMIT_FILE:
            over.append(f"file {max_path.relative_to(ROOT)}={max_file}")
        for directory in sorted({f.parent for f in files}):
            count = len(dir_source_files(directory))
            if count > LIMIT_DIR:
                over.append(f"dir {directory.relative_to(ROOT)}={count}")
        if over:
            print(
                "  OVER (not enforced; outside the local trusted-checking perimeter): "
                + ", ".join(over)
            )
        else:
            print("  within limits")

    print("[external, not counted] Lean kernel, Lean/Mathlib, Charon/Aeneas pins: proof/README.md")

    if failures:
        print(f"\nFAIL: {len(failures)} assurance breach(es)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nOK: every bounded population is defined and within limits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
