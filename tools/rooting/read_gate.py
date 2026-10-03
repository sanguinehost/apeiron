#!/usr/bin/env python3
"""The §1 reading-order gate.

`AGENTS.md` §1 states a reading order over the governing documents. Before this
gate that order named no enforcer, so under §1a it was non-binding prose — an
agent was *expected* to read `docs/direction.md` and the rest, and nothing made
that true or false. This gate makes the order binding in the two ways §1 allows:

  * **referenced in `AGENTS.md`.** A stage document holds information an agent
    must act on, and `AGENTS.md` is the one file every agent is delivered. So a
    stage document that is not referenced in `AGENTS.md` is a finding: either
    inline what it carries, or reference it. `--check` reddens by name.
  * **a dispatch receipt.** A dispatch carries the sha256 of each stage document,
    so the identity the agent worked against is pinned. `--receipt` writes it;
    `--verify` re-checks it against the tree and reddens by name on drift.

What this does **not** establish (obligation 11 trust boundary): that a model
read or understood a document. A receipt pins identity and detects drift; the
host delivers `AGENTS.md` and the dispatch prompt. Comprehension is not
observable from here and is not claimed.

Usage:
    python3 tools/rooting/read_gate.py --check
    python3 tools/rooting/read_gate.py --receipt /tmp/opencode/read-receipt.json --id APEIRON-...
    python3 tools/rooting/read_gate.py --verify /tmp/opencode/read-receipt.json
    python3 tools/rooting/read_gate.py --self-test
    python3 tools/rooting/read_gate.py --controls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

EXIT_OK = 0
EXIT_REJECT = 1
EXIT_USAGE = 2

AGENTS = "AGENTS.md"

# The §1 stage documents, in reading order. `AGENTS.md` is the delivered file
# itself and is exempt from the "referenced in AGENTS.md" rule.
STAGE_DOCS: tuple[str, ...] = (
    "AGENTS.md",
    "docs/direction.md",
    "docs/kernel-boundary.md",
    "docs/plan.md",
    "proof/README.md",
    "README.md",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan(root: Path) -> list[str]:
    """Findings for the tree at ``root``. Empty is acceptance."""
    agents = root / AGENTS
    if not agents.exists():
        return [f"missing {AGENTS}"]
    text = agents.read_text(encoding="utf-8", errors="replace")
    findings: list[str] = []
    for doc in STAGE_DOCS:
        path = root / doc
        if not path.exists():
            findings.append(f"stage document missing: {doc}")
            continue
        if doc != AGENTS and doc not in text:
            findings.append(f"stage document not referenced in {AGENTS}: {doc}")
    return findings


def make_receipt(root: Path, dispatch_id: str) -> dict:
    docs = []
    for doc in STAGE_DOCS:
        path = root / doc
        if not path.exists():
            raise SystemExit(f"read_gate: stage document missing: {doc}")
        docs.append({"path": doc, "sha256": digest(path)})
    return {"schema": 1, "id": dispatch_id, "docs": docs}


def verify_receipt(root: Path, receipt: dict) -> list[str]:
    findings: list[str] = []
    for entry in receipt.get("docs", []):
        name = entry.get("path", "")
        path = root / name
        if not path.exists():
            findings.append(f"receipt names a missing document: {name}")
            continue
        actual = digest(path)
        if actual != entry.get("sha256"):
            findings.append(
                f"digest mismatch: {name} receipt {str(entry.get('sha256'))[:12]} "
                f"on disk {actual[:12]}"
            )
    return findings


# ---------------------------------------------------------------------------
# Fixtures: --self-test and --controls materialise a synthetic tree; the working
# tree is never mutated.
# ---------------------------------------------------------------------------

_BODY = "synthetic stage document\n"


def _materialize(root: Path) -> None:
    agents = "\n".join(f"- {doc}: {_BODY.strip()}" for doc in STAGE_DOCS)
    (root / AGENTS).write_text(agents + "\n")
    for doc in STAGE_DOCS:
        if doc == AGENTS:
            continue
        path = root / doc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_BODY)


def _case(name: str, seed, expect: str, needle: str) -> tuple[str, object, str, str]:
    return (name, seed, expect, needle)


def self_test() -> int:
    failures: list[str] = []
    base = Path(tempfile.mkdtemp(prefix="read-gate-clean-"))
    try:
        _materialize(base)
        if scan(base):
            print(f"SELF-TEST FAIL clean fixture: {scan(base)[0]}")
            return EXIT_REJECT
        print("SELF-TEST GREEN clean fixture")

        def unfix_missing_path(root: Path) -> None:
            (root / "docs/plan.md").unlink()

        def unfix_unreferenced(root: Path) -> None:
            text = (root / AGENTS).read_text().replace("docs/kernel-boundary.md", "KB")
            (root / AGENTS).write_text(text)

        cases = [
            _case("missing stage document", unfix_missing_path, "reject", "stage document missing: docs/plan.md"),
            _case("stage document not referenced", unfix_unreferenced, "reject", "not referenced in AGENTS.md: docs/kernel-boundary.md"),
        ]
        for name, seed, expect, needle in cases:
            work = Path(tempfile.mkdtemp(prefix="read-gate-case-"))
            try:
                shutil.copytree(base, work, dirs_exist_ok=True)
                seed(work)
                findings = scan(work)
                if expect == "reject":
                    if not findings:
                        failures.append(f"{name}: survived, expected rejection")
                        print(f"SELF-TEST FAIL {name}: survived")
                    elif not any(needle in f for f in findings):
                        failures.append(f"{name}: rejected by {findings}, expected {needle!r}")
                        print(f"SELF-TEST FAIL {name}: rejected by {findings}")
                    else:
                        print(f"SELF-TEST RED  {name}: {next(f for f in findings if needle in f)}")
                elif findings:
                    failures.append(f"{name}: rejected, expected acceptance")
                    print(f"SELF-TEST FAIL {name}: {findings[0]}")
                else:
                    print(f"SELF-TEST GREEN {name}")
            finally:
                shutil.rmtree(work, ignore_errors=True)

        # The receipt half: a stale receipt must redden by name.
        work = Path(tempfile.mkdtemp(prefix="read-gate-receipt-"))
        try:
            _materialize(work)
            rec = make_receipt(work, "SELF-TEST")
            (work / "docs/direction.md").write_text("drifted\n")
            findings = verify_receipt(work, rec)
            if not any("digest mismatch: docs/direction.md" in f for f in findings):
                failures.append("stale receipt: survived, expected digest mismatch")
                print("SELF-TEST FAIL stale receipt: survived")
            else:
                print(f"SELF-TEST RED  stale receipt: {findings[0]}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if failures:
        print(f"SELF-TEST FAIL: {len(failures)} case(s) failed")
        return EXIT_REJECT
    print("SELF-TEST PASS: clean fixture accepted, 3 rejection fixture(s) rejected")
    return EXIT_OK


def controls() -> int:
    root = Path(tempfile.mkdtemp(prefix="read-gate-controls-"))
    try:
        _materialize(root)
        baseline = scan(root)
        if baseline:
            print(f"CONTROL FAIL baseline: expected green, got {baseline[0]}")
            return EXIT_REJECT
        print("CONTROL pass baseline: green")

        rec = make_receipt(root, "CONTROL")
        original = (root / "docs/direction.md").read_text()
        (root / "docs/direction.md").write_text("mutated\n")
        drift = verify_receipt(root, rec)
        if not any("digest mismatch: docs/direction.md" in f for f in drift):
            print("CONTROL FAIL mutation: receipt did not redden on drift")
            return EXIT_REJECT
        print(f"CONTROL pass mutation: red by name ({drift[0]})")

        (root / "docs/direction.md").write_text(original)
        if verify_receipt(root, rec) or scan(root):
            print("CONTROL FAIL restore: expected green after byte-identical restore")
            return EXIT_REJECT
        print("CONTROL pass restore: green")
        return EXIT_OK
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the AGENTS.md §1 reading-order gate")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify stage docs resolve and are referenced in AGENTS.md")
    mode.add_argument("--receipt", metavar="OUT", help="write a dispatch receipt of stage-document digests")
    mode.add_argument("--verify", metavar="RECEIPT", help="verify a receipt against the tree")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--controls", action="store_true")
    parser.add_argument("--id", default="", help="dispatch id recorded in the receipt")
    parser.add_argument("--root", default=".", help="repository root (default: cwd)")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.controls:
        return controls()

    root = Path(args.root).resolve()
    if args.receipt:
        rec = make_receipt(root, args.id)
        out = Path(args.receipt)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2) + "\n")
        manifest = hashlib.sha256(
            "".join(f"{d['path']}\0{d['sha256']}\n" for d in rec["docs"]).encode()
        ).hexdigest()
        print(f"read_gate: receipt {out} id={args.id or '-'} docs={len(rec['docs'])} manifest={manifest[:16]}")
        return EXIT_OK
    if args.verify:
        rec = json.loads(Path(args.verify).read_text())
        findings = verify_receipt(root, rec)
        if findings:
            for finding in findings:
                print(f"read_gate: REFUSED {finding}", file=sys.stderr)
            return EXIT_REJECT
        print(f"read_gate: receipt {args.verify} matches the tree ({len(rec.get('docs', []))} document(s))")
        return EXIT_OK

    findings = scan(root)
    for doc in STAGE_DOCS:
        path = root / doc
        if path.exists():
            print(f"  OK   {doc} sha256={digest(path)}")
    if findings:
        for finding in findings:
            print(f"read_gate: REFUSED {finding}", file=sys.stderr)
        return EXIT_REJECT
    print("read_gate: every stage document resolves and is referenced in AGENTS.md")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
