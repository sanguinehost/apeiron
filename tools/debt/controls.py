#!/usr/bin/env python3
"""Negative controls for the debt detector: seed a defect, watch the class fire.

Per AGENTS.md §4 obligation 7 a check that cannot fail has no weight, and per §3
a mutation control edits a throwaway copy, never the working tree. Every control
here builds a fixture repository under `--scratch`, seeds one anomaly, runs
`debt_scan.py` against the fixture, and asserts the named class reports it; then
it asserts the class is silent on the same fixture with the defect removed.

Run:
    python3 tools/debt/controls.py [--scratch /home/socol/oc-tmp/debt-ctl] [--keep]

Exit 0 iff every control both fires on its seeded defect and is quiet without it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "debt_scan.py")


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def sha256_of(path):
    import hashlib

    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def make_fixture(root):
    """A miniature repository with one evidence receipt and one digest claim."""
    os.makedirs(root, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "ctl@twn.systems")
    git(root, "config", "user.name", "controls")
    write(os.path.join(root, "src/lib.rs"), "pub fn f() -> u8 { 1 }\n")
    write(os.path.join(root, "docs/evidence/receipt.md"), "# receipt\n\n`cargo test` rc 0.\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "seed")
    good = sha256_of(os.path.join(root, "docs/evidence/receipt.md"))
    write(
        os.path.join(root, "docs/evidence/ledger.jsonl"),
        json.dumps({"id": "receipt", "kind": "receipt",
                    "path": "docs/evidence/receipt.md", "sha256": good,
                    "summary": "`cargo test` rc 0"}) + "\n",
    )
    git(root, "add", "-A")
    git(root, "commit", "-qm", "ledger")
    return root


def board_capture(root, rows, name="board"):
    """Write a board capture in the REST tasks/list shape and return its path.

    `name` must differ per capture: two captures at one path would make the
    "before" scan read whatever the second capture wrote, and the control would
    compare the clean board against itself.
    """
    p = os.path.join(root, f"{name}.json")
    with open(p, "w") as fh:
        json.dump({"tasks": rows}, fh)
    return p


def run_scan(root, board_file):
    cmd = [sys.executable, SCAN, "--repo", root, "--json"]
    if board_file:
        cmd += ["--board", board_file]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise SystemExit(f"scan failed rc={p.returncode}\n{p.stderr}")
    return json.loads(p.stdout)


def row(rid, name, description="", status="in_progress"):
    """One board row in the REST shape, description as ProseMirror doc JSON."""
    doc = {"content": [{"content": [{"text": description, "type": "text"}], "type": "paragraph"}]}
    return {
        "id": rid, "name": name, "description": json.dumps(doc),
        "status": {"value": status, "label": status, "closed": status == "done"},
        "updated_at": "2026-10-01T00:00:00Z", "inserted_at": "2026-10-01T00:00:00Z",
        "comments_count": 0,
    }


def control_d_digest_drift(root):
    """Seeded defect: a file the ledger claims changes; its recorded digest does not.

    The defect must be seeded on the *claimed* path. An earlier version of this
    control mutated `src/lib.rs`, which no digest record names, and correctly
    did not fire -- a control whose seed is outside the class's population
    proves nothing.
    """
    out = {}
    before = run_scan(root, None)
    out["quiet_before"] = len(before["classes"]["D"]["primary"]["drifted"]) == 0
    write(os.path.join(root, "docs/evidence/receipt.md"), "# receipt\n\n`cargo test` rc 1.\n")
    after = run_scan(root, None)
    drifted = after["classes"]["D"]["primary"]["drifted"]
    out["fires"] = any(d["path"] == "docs/evidence/receipt.md" for d in drifted)
    out["count"] = len(drifted)
    return out


def control_d_restore(root):
    """Restore: the recorded digest matches again and the class is silent."""
    before = run_scan(root, None)
    n = len(before["classes"]["D"]["primary"]["drifted"])
    git(root, "checkout", "--", "docs/evidence/receipt.md")
    after = run_scan(root, None)
    return {"fired": n > 0, "silent_after_restore": len(after["classes"]["D"]["primary"]["drifted"]) == 0}


def control_f_orphan(root):
    """Seeded defect: a receipt file with no index row."""
    before = run_scan(root, None)
    write(os.path.join(root, "docs/evidence/orphan.md"), "# orphan receipt\n")
    after = run_scan(root, None)
    return {
        "quiet_before": len(before["classes"]["F"]["unindexed"]) == 0,
        "fires": "docs/evidence/orphan.md" in after["classes"]["F"]["unindexed"],
    }


def control_h_missing_declaration(root):
    """Seeded defect: a commit with no `board:` line."""
    write(os.path.join(root, "src/lib.rs"), "pub fn f() -> u8 { 3 }\n")
    git(root, "commit", "-qam", "undeclared work")
    after = run_scan(root, None)
    missing = after["classes"]["H"]["commits_without_declaration"]
    return {"fires": any(m["subject"].startswith("undeclared work") for m in missing)}


def control_h_present_declaration(root):
    """Positive: declaring `none -- housekeeping` silences it, as the hook allows."""
    write(os.path.join(root, "docs/evidence/housekeeping.md"), "# housekeeping\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "housekeeping sweep\n\nboard: none -- housekeeping\n")
    after = run_scan(root, None)
    missing = after["classes"]["H"]["commits_without_declaration"]
    return {"silent": not any(m["subject"].startswith("housekeeping sweep") for m in missing)}


def control_c_dangling(root):
    """Seeded defect: a row naming a path that does not resolve."""
    board = board_capture(
        root, [row("ctl-1", "CTL-1: point at a missing source",
                   "See `src/does_not_exist.rs` for the body.")]
    )
    clean = [row("ctl-1", "CTL-1: point at a real source", "See `src/lib.rs` for the body.")]
    before = run_scan(root, board_capture(root, clean, name="board-clean"))
    after = run_scan(root, board)
    paths = [p for r in after["classes"]["C"]["rows"] for p in r["paths"]]
    return {
        "quiet_before": len(before["classes"]["C"]["rows"]) == 0,
        "fires": "src/does_not_exist.rs" in paths,
    }


def control_b_duplicate(root):
    """Seeded defects at a population where the non-match distribution exists.

    An earlier version used four rows. That cannot test near-duplicate
    sensitivity: with four rows every pair shares its leading two-token prefix, so
    u(prefix) estimated near 0.5 and the Fellegi-Sunter posterior correctly fell
    below 0.5 for a genuine near-duplicate. The method was right and the fixture
    was too small. At 40 rows with distinct leading prefixes u is estimable, and
    both the exact and the one-word near-duplicate are recovered while the
    unrelated pair is not reported.
    """
    rng = random.Random(4242)
    rows, names = [], {}
    adjectives = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                  "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi"]
    nouns = ["harness", "ladder", "bridge", "kernel", "probe", "ledger", "census",
             "refiner", "archive", "frontier", "residue", "beacon", "seam", "anchor"]
    verbs = ["repair", "rebuild", "audit", "extend", "regress", "fold", "split", "merge"]
    seen = set()
    for i in range(36):
        while True:
            a = rng.choice(adjectives)
            n = rng.choice(nouns)
            v = rng.choice(verbs)
            if (a, n, v) not in seen:
                seen.add((a, n, v))
                break
        rid = f"ctl-{i:02d}"
        nm = f"{a}-{n}-{v}: a scoped task number {i}"
        names[rid] = nm
        rows.append(row(rid, nm, f"Gate: `cargo test` lane {i}. Source: src/lib.rs."))
    # Seeded duplicate: one row copied verbatim under a new id.
    rows.append(row("ctl-exact", names["ctl-00"], rows[0]["description"]))
    names["ctl-exact"] = names["ctl-00"]
    # Seeded near-duplicate: the same row with one word changed.
    near = names["ctl-01"].replace("repair", "repairs")
    rows.append(row("ctl-near", near, rows[1]["description"]))
    names["ctl-near"] = near
    # Seeded non-match: shares the prefix, differs in every other word.
    unrelated = f"{names['ctl-01'].split(':')[0].split('-')[0]}-{rng.choice(nouns)}-fold: unrelated scope entirely"
    rows.append(row("ctl-unrelated", unrelated, "Gate: `lake build` only."))
    names["ctl-unrelated"] = unrelated

    out = run_scan(root, board_capture(root, rows, name="board-dup"))["classes"]["B"]["link"]
    pairs = {(p["a"], p["b"]) for p in out["pairs"]}
    return {
        "n_rows": len(rows),
        "u": out["u"],
        "identified": out["identified"],
        "exact_pair_found": ("ctl-00", "ctl-exact") in pairs,
        "near_pair_found": ("ctl-01", "ctl-near") in pairs,
        "no_false_pair": ("ctl-01", "ctl-unrelated") not in pairs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/home/socol/oc-tmp/debt-ctl")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    root = os.path.join(args.scratch, "fixture")
    if os.path.exists(root):
        shutil.rmtree(root)
    make_fixture(root)

    results = {}
    results["d_quiet_when_clean"] = control_d_digest_drift(root)
    results["d_restore"] = control_d_restore(root)
    results["f_orphan"] = control_f_orphan(root)
    results["c_dangling"] = control_c_dangling(root)
    results["b_duplicate"] = control_b_duplicate(root)
    results["h_missing_declaration"] = control_h_missing_declaration(root)
    results["h_present_declaration"] = control_h_present_declaration(root)

    expect_fire = ("d_quiet_when_clean.fires", "d_restore.fired", "f_orphan.fires",
                   "c_dangling.fires", "b_duplicate.exact_pair_found",
                   "b_duplicate.near_pair_found", "h_missing_declaration.fires")
    expect_quiet = ("d_quiet_when_clean.quiet_before", "d_restore.silent_after_restore",
                    "f_orphan.quiet_before", "c_dangling.quiet_before",
                    "b_duplicate.no_false_pair", "h_present_declaration.silent")

    def get(d, dotted):
        for part in dotted.split("."):
            d = d[part]
        return d

    failures = []
    for k in expect_fire:
        if not get(results, k):
            failures.append(f"{k} did not fire")
    for k in expect_quiet:
        if not get(results, k):
            failures.append(f"{k} was not quiet")

    print(json.dumps(results, indent=1))
    print("-" * 60)
    for k in sorted(set(expect_fire) | set(expect_quiet)):
        kind = "FIRE " if k in expect_fire else "QUIET"
        print(f"{kind} {k:44} {get(results, k)}")
    print("-" * 60)
    if failures:
        for f in failures:
            print(f"CONTROL FAIL: {f}")
        print(f"CHECK FAIL: {len(failures)} control(s) did not discriminate")
        return 1
    print("CHECK PASS: every control fired on its seeded defect and was quiet without it")
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())