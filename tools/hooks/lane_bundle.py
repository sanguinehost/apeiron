#!/usr/bin/env python3
# Row: `apeiron-harness-FiQJ8KYxibPFz1lDzxuAao` (APEIRON-HARNESS-DURABILITY).
"""lane-bundle: make an uncommitted lane's work recoverable before it can be pruned.

THE FAILURE THIS ANSWERS
------------------------
OpenCode gives a detached worktree no protection and no warning. A lane's
uncommitted diff lives only inside that worktree until someone commits or
copies it out; an external `git worktree prune`, a `spinoff` sweep or a /tmp
cleanup can delete it mid-task, with no notification delivered to the lane.
The canonical example is /home/socol/oc-tmp/proof-land, destroyed twice:

  1. The historical cleanup/prune that took proof-land together with vv-v1a,
     vv-v1b, readloop-repair and v4-cost. Those worktrees held uncommitted
     diffs; no bundle existed, so the work was simply gone.
  2. The mid-task deletion this cycle, observed as `git worktree list` falling
     from 16 entries to 4 while a lane was still working. The same sweep
     removed 83 worktrees at once, proof-land among them, while its lane was
     mid-task. Nothing in the worktree could have seen it coming.

Both are one event: work that existed only inside a worktree, deleted from
outside the worktree. `lane-bundle` copies that work into a declared bundle
directory before it can be pruned:

    /home/socol/oc-tmp/bundles/<name>-<utc>.<content12>.patch
    /home/socol/oc-tmp/bundles/<name>-<utc>.<content12>.tar
    /home/socol/oc-tmp/bundles/<name>-<utc>.<content12>.json   (manifest)

The patch is `git diff --binary <base>` for the tracked change; the tar holds
every untracked (non-ignored) file with its mode; the manifest records the
worktree, its base and head commits, and the sha256 of each artefact. The
sha256 of each produced file is printed.

TRUST BOUNDARY
--------------
This cannot prevent an external `git worktree prune` or `rm -rf`; nothing
running inside the worktree can. It only makes the loss recoverable: a bundle
taken before the deletion restores the tracked diff (`git apply`) and the
untracked files (`tar -xf`) byte-exactly into a fresh checkout of the recorded
base. It does not protect work performed after the last bundle. It never
modifies the worktree and never deletes anything. It is idempotent: identical
content converges on one bundle file, which is reused rather than duplicated,
so re-running is a no-op. Ignored files (those matched by .gitignore, e.g.
target/ and proof/.lake/) are NOT captured -- they are build products, not lane
work, and bundling them would defeat the tool's cheapness; the inventory
reports their presence so the choice is visible.

USAGE
-----
    lane_bundle.py <worktree> [<worktree> ...] [--base <rev>] [--out DIR]
    lane_bundle.py --all [<repo>] [--root DIR] [--out DIR]

Exit status is 0 on success (including "clean, nothing to bundle"), 2 on error.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUT = Path(os.environ.get("APEIRON_BUNDLE_DIR", "/home/socol/oc-tmp/bundles"))
DEFAULT_ROOT = Path(os.environ.get("APEIRON_LANE_ROOT", "/home/socol/oc-tmp"))
REF_CANDIDATES = ("main", "origin/main", "master")
TOOL_ID = "lane-bundle/v1"


class BundleError(RuntimeError):
    """A git command the bundle depends on did not succeed."""


def git_env() -> dict:
    env = dict(os.environ)
    # Never let a read-only inspection write to the worktree's index.
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return env


def git(wt, *args, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", "-C", str(wt), *args],
        capture_output=True,
        env=git_env(),
    )
    if check and proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise BundleError(f"git -C {wt} {' '.join(args)} failed rc={proc.returncode}: {msg}")
    return proc


def rev_parse(wt, rev: str):
    proc = git(wt, "rev-parse", "--verify", "--quiet", rev, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.decode().strip() or None


def resolve_base(wt, ref: str | None = None):
    """The rev a bundle diffs against, and the ref it was resolved from.

    Prefers merge-base(HEAD, main) so a lane that has committed on top of its
    fork point still has those commits captured by `git diff <base>`. Falls
    back to HEAD (the uncommitted-change case) when no integration ref exists.
    """
    head = rev_parse(wt, "HEAD") or ""
    for cand in ((ref,) if ref else REF_CANDIDATES):
        if rev_parse(wt, cand) is None:
            continue
        mb = git(wt, "merge-base", "HEAD", cand, check=False)
        if mb.returncode == 0:
            return mb.stdout.decode().strip(), cand
    return head, None


def head_sha(wt) -> str:
    return rev_parse(wt, "HEAD") or ""


def untracked(wt):
    proc = git(wt, "ls-files", "--others", "--exclude-standard", "-z")
    return [x for x in proc.stdout.decode("utf-8", "surrogateescape").split("\0") if x]


def ignored_present(wt) -> bool:
    proc = git(wt, "ls-files", "--others", "--ignored", "--exclude-standard", "-z", check=False)
    return len(proc.stdout) > 0


def patch_bytes(wt, base: str) -> bytes:
    proc = git(wt, "diff", "--binary", "--no-color", "--no-ext-diff", base, "--")
    return proc.stdout


def build_tar(wt, rels) -> bytes:
    """A deterministic tar of the given worktree-relative paths.

    Metadata is normalised (uid/gid/names zeroed, mtime zeroed) so an unchanged
    tree yields byte-identical bytes across runs, which is what makes the
    idempotence check work and the sha stable.
    """
    rels = sorted(set(rels))
    if not rels:
        return b""
    buf = io.BytesIO()

    def norm(ti):
        ti.uid = 0
        ti.gid = 0
        ti.uname = ""
        ti.gname = ""
        ti.mtime = 0
        return ti

    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for rel in rels:
            src = Path(wt) / rel
            if not (src.exists() or src.is_symlink()):
                continue
            tf.add(str(src), arcname=rel, recursive=False, filter=norm)
    return buf.getvalue()


def combined_digest(base: str, patch: bytes, tar: bytes) -> str:
    h = hashlib.sha256()
    h.update(TOOL_ID.encode())
    h.update(b"\0")
    h.update(base.encode())
    h.update(b"\0")
    h.update(len(patch).to_bytes(8, "big"))
    h.update(patch)
    h.update(len(tar).to_bytes(8, "big"))
    h.update(tar)
    return h.hexdigest()


def sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return s or "worktree"


def load_manifest(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def find_existing(out: Path, name: str, tag: str, wt: Path):
    for man in sorted(out.glob(f"{name}-*.{tag}.json")):
        m = load_manifest(man)
        if m and m.get("worktree") == str(wt) and m.get("content_sha256"):
            m["manifest_path"] = str(man)
            return m
    return None


def print_bundle(man: dict, manifest_path: str | None = None) -> None:
    for key in ("patch", "tar"):
        d = man.get(key)
        if d:
            print(f"  {key:8s}: {d['path']}  sha256={d['sha256']}  bytes={d['bytes']}")
    path = manifest_path or man.get("manifest_path")
    if path:
        print(f"  manifest: {path}")


def bundle_one(wt, out, base_ref: str | None = None, quiet: bool = False) -> dict:
    wt = Path(wt).resolve()
    out = Path(out)
    if not wt.is_dir():
        raise BundleError(f"not a worktree directory: {wt}")
    base, ref = resolve_base(wt, base_ref)
    head = head_sha(wt)
    patch = patch_bytes(wt, base)
    rels = untracked(wt)
    tar = build_tar(wt, rels)
    ign = ignored_present(wt)

    if not patch and not tar:
        if not quiet:
            print(f"{wt}: clean - nothing to bundle")
            if ign:
                print("  note: ignored files present (not captured)")
        return {"worktree": str(wt), "state": "clean", "ignored_present": ign}

    dig = combined_digest(base, patch, tar)
    tag = dig[:12]
    name = sanitize(wt.name)
    out.mkdir(parents=True, exist_ok=True)

    existing = find_existing(out, name, tag, wt)
    if existing is not None:
        if not quiet:
            print(f"{wt}: [idempotent] identical content already bundled; reusing")
            print_bundle(existing, existing.get("manifest_path"))
        return existing

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    patch_path = out / f"{name}-{stamp}.{tag}.patch"
    tar_path = out / f"{name}-{stamp}.{tag}.tar"
    man_path = out / f"{name}-{stamp}.{tag}.json"

    man = {
        "tool": TOOL_ID,
        "worktree": str(wt),
        "name": name,
        "base": base,
        "base_ref": ref,
        "head": head,
        "created_utc": stamp,
        "content_sha256": dig,
        "ignored_present": ign,
        "untracked_count": len(rels),
        "manifest_path": str(man_path),
    }
    if patch:
        patch_path.write_bytes(patch)
        man["patch"] = {
            "path": str(patch_path),
            "sha256": hashlib.sha256(patch).hexdigest(),
            "bytes": len(patch),
        }
    if tar:
        tar_path.write_bytes(tar)
        man["tar"] = {
            "path": str(tar_path),
            "sha256": hashlib.sha256(tar).hexdigest(),
            "bytes": len(tar),
        }
    man_path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")

    if not quiet:
        base_note = f" via {ref}" if ref else ""
        print(f"{wt}: bundled (base {base[:12]}{base_note}, head {head[:12]})")
        print_bundle(man, str(man_path))
        if ign:
            print("  note: ignored files present (not captured)")
    return man


def list_worktrees(repo):
    proc = git(repo, "worktree", "list", "--porcelain")
    entries = []
    cur = {}
    for raw in proc.stdout.decode("utf-8", "surrogateescape").splitlines():
        if not raw.strip():
            if cur:
                entries.append(cur)
                cur = {}
            continue
        key, _, val = raw.partition(" ")
        if key == "worktree":
            cur["path"] = val
        elif key == "HEAD":
            cur["head"] = val
        elif key == "branch":
            cur["branch"] = val
        elif key == "detached":
            cur["detached"] = True
        elif key == "bare":
            cur["bare"] = True
        elif key == "prunable":
            cur["prunable"] = val or True
        elif key == "locked":
            cur["locked"] = val or True
    if cur:
        entries.append(cur)
    return entries


def is_main(wt) -> bool:
    """True for the primary working tree, whose git-dir is the common dir."""
    wt = Path(wt)
    gd = git(wt, "rev-parse", "--git-dir", check=False).stdout.decode().strip()
    cd = git(wt, "rev-parse", "--git-common-dir", check=False).stdout.decode().strip()
    if not gd or not cd:
        return False
    rd = (wt / gd).resolve() if not os.path.isabs(gd) else Path(gd)
    rc = (wt / cd).resolve() if not os.path.isabs(cd) else Path(cd)
    return rd == rc


def under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Bundle a lane worktree's uncommitted diff and untracked files, "
        "non-destructively and idempotently.",
    )
    ap.add_argument("worktrees", nargs="*", help="worktree paths (or the repo for --all)")
    ap.add_argument("--all", action="store_true", help="bundle every linked worktree under --root")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help=f"root for --all (default {DEFAULT_ROOT})")
    ap.add_argument("--base", default=None, help="base rev to diff against (default: merge-base with main)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"bundle directory (default {DEFAULT_OUT})")
    ap.add_argument("--json", action="store_true", help="emit manifests as JSON")
    args = ap.parse_args(argv)

    if args.all:
        repo = args.worktrees[0] if args.worktrees else os.getcwd()
        root = Path(args.root).resolve()
        targets = []
        for e in list_worktrees(repo):
            if e.get("bare"):
                continue
            p = Path(e["path"]).resolve()
            if is_main(p) or not under(p, root):
                continue
            targets.append(p)
    else:
        if not args.worktrees:
            ap.error("give at least one worktree path, or --all")
        targets = [Path(p) for p in args.worktrees]

    results = []
    try:
        for wt in targets:
            results.append(bundle_one(wt, args.out, base_ref=args.base))
    except BundleError as exc:
        print(f"lane-bundle: error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
