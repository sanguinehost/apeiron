#!/usr/bin/env python3
# Row: `apeiron-harness-FiQJ8KYxibPFz1lDzxuAao` (APEIRON-HARNESS-DURABILITY).
"""lane-inventory: list every worktree, its base, its dirty files and its bundle.

THE FAILURE THIS ANSWERS
------------------------
Same failure as lane_bundle.py, seen from the other side. An external
`git worktree prune`, a `spinoff` sweep or a /tmp cleanup deletes detached
worktrees with no warning; the canonical example is /home/socol/oc-tmp/
proof-land, destroyed twice: (1) the historical cleanup/prune that took
proof-land with vv-v1a, vv-v1b, readloop-repair and v4-cost, and (2) the
mid-task deletion this cycle that dropped `git worktree list` from 16 entries
to 4 and removed 83 worktrees at once. Before any prune, a lane needs to see
which worktrees hold uncommitted work and whether a bundle already covers it.

`lane-inventory` prints, per worktree: the resolved base commit, head, whether
it is dirty or clean, the sha256 of each dirty file, and whether a bundle
exists and is fresh (its content digest equals the worktree's current content)
or stale (a bundle exists but the worktree has moved on). A prune can then be
preceded by a check rather than a surprise.

TRUST BOUNDARY
--------------
This is a read-only report. It cannot prevent an external prune, and it does
not delete, move or modify anything. It never modifies a worktree. Its
freshness test regenerates the same digest lane_bundle.py uses, so a "fresh"
row means the current tracked diff and untracked set are byte-identical to a
stored bundle; it says nothing about work done after that bundle. Ignored
files (build products) are excluded from the digest and from the dirty list;
they are reported only as a note in the JSON.

USAGE
-----
    lane_inventory.py [<repo>] [--root DIR] [--out DIR] [--json]

Exit status is 0 on success, 2 on error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lane_bundle import (  # noqa: E402
    DEFAULT_OUT,
    BundleError,
    build_tar,
    combined_digest,
    git,
    head_sha,
    ignored_present,
    is_main,
    list_worktrees,
    load_manifest,
    patch_bytes,
    resolve_base,
    under,
    untracked,
)
from lane_bundle import DEFAULT_ROOT  # noqa: E402


def file_sha(wt: Path, rel: str) -> str:
    p = wt / rel
    if p.is_symlink():
        return "symlink:" + os.readlink(p)
    if not p.exists():
        return "(deleted)"
    if p.is_dir():
        return "(dir)"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status_entries(wt):
    """Parse `git status --porcelain -z` into (XY, path) pairs."""
    proc = git(wt, "status", "--porcelain", "-z", "--untracked-files=all", check=False)
    fields = proc.stdout.split(b"\0")
    out = []
    i = 0
    while i < len(fields):
        f = fields[i]
        if not f:
            i += 1
            continue
        xy = f[:2].decode("utf-8", "replace")
        path = f[3:].decode("utf-8", "surrogateescape")
        out.append((xy, path))
        if xy[:1] in ("R", "C"):
            i += 1  # the rename/copy source path is a following NUL field
        i += 1
    return out


def bundle_status(wt: Path, out: Path, content_sha: str | None) -> dict:
    if not out.is_dir():
        return {"state": "none"}
    latest = None
    latest_path = None
    for man_path in sorted(out.glob("*.json")):
        m = load_manifest(man_path)
        if not m or m.get("worktree") != str(wt):
            continue
        latest = m
        latest_path = man_path
        if content_sha and m.get("content_sha256") == content_sha:
            return {
                "state": "fresh",
                "manifest": str(man_path),
                "created_utc": m.get("created_utc"),
            }
    if latest is None:
        return {"state": "none"}
    return {
        "state": "stale",
        "manifest": str(latest_path),
        "created_utc": latest.get("created_utc"),
    }


def inventory_one(wt, out: Path) -> dict:
    wt = Path(wt).resolve()
    base, ref = resolve_base(wt)
    head = head_sha(wt)
    entries = status_entries(wt)

    dirty = []
    modified = 0
    untracked_count = 0
    for xy, path in entries:
        if xy == "??":
            untracked_count += 1
        else:
            modified += 1
        dirty.append({"status": xy, "path": path, "sha256": file_sha(wt, path)})

    patch = patch_bytes(wt, base)
    tar = build_tar(wt, untracked(wt))
    if patch or tar:
        content_sha = combined_digest(base, patch, tar)
    else:
        content_sha = None

    return {
        "worktree": str(wt),
        "main": is_main(wt),
        "head": head,
        "base": base,
        "base_ref": ref,
        "state": "dirty" if dirty else "clean",
        "modified": modified,
        "untracked": untracked_count,
        "ignored_present": ignored_present(wt),
        "dirty_files": dirty,
        "content_sha256": content_sha,
        "bundle": bundle_status(wt, out, content_sha),
    }


def emit_human(entries) -> None:
    for e in entries:
        label = " (main)" if e["main"] else ""
        base_note = f" via {e['base_ref']}" if e["base_ref"] else ""
        print(f"{e['worktree']}{label}")
        print(f"  base  : {e['base'][:12]}{base_note}")
        print(f"  head  : {e['head'][:12]}")
        print(f"  state : {e['state']} ({e['modified']} modified, {e['untracked']} untracked)")
        b = e["bundle"]
        if b["state"] == "none":
            print("  bundle: none")
        else:
            print(f"  bundle: {b['state']}  {b.get('manifest', '')}")
        for d in e["dirty_files"]:
            print(f"    {d['status']}  {d['path']}  sha256={d['sha256']}")
        if e["ignored_present"]:
            print("  note  : ignored files present (not in digest/dirty list)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Inventory worktrees: base, dirty files with sha256, and bundle state.",
    )
    ap.add_argument("repo", nargs="?", default=os.getcwd(), help="any worktree of the repo")
    ap.add_argument("--root", default=None, help=f"only worktrees under this root (default all; e.g. {DEFAULT_ROOT})")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"bundle directory (default {DEFAULT_OUT})")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    out = Path(args.out)
    root = Path(args.root).resolve() if args.root else None

    try:
        worktrees = list_worktrees(args.repo)
    except BundleError as exc:
        print(f"lane-inventory: error: {exc}", file=sys.stderr)
        return 2

    selected = []
    for e in worktrees:
        if e.get("bare"):
            continue
        p = Path(e["path"]).resolve()
        if root is not None and not under(p, root):
            continue
        selected.append(p)

    results = []
    try:
        for wt in selected:
            results.append(inventory_one(wt, out))
    except BundleError as exc:
        print(f"lane-inventory: error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        emit_human(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
