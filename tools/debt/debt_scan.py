#!/usr/bin/env python3
"""APEIRON-PLATFORM-DEBT-DETECTOR: eight technical-debt and data-anomaly classes.

Read-only. Collects the board, the repositories, the evidence ledger and the
live-lane table, then prints a per-class anomaly report: count plus top
instances. Every number printed here is the output of a command named in the
report and of the code in this file; nothing is carried over from prose.

Usage:
    python3 tools/debt/debt_scan.py [--repo .] [--project ID] [--top N] [--json]

Data sources, and what each one cannot see, are stated in TRUST BOUNDARY at the
bottom of every run.

Classes
    A stale in-progress      row In progress, no commit naming it, no lane
    B duplicate rows         record linkage over name+description
    C dangling reference     row names a path/symbol that does not resolve
    D digest drift           recorded sha256 != sha256 of the file it names
    E unrefreshed evidence   receipt predates a source it names
    F orphaned receipt       evidence file with no index row (both directions)
    G unanchored git state   worktree/branch no row names; row no worktree
    H undeclared commit      commit with no `board:` line; row that is gone
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from similarity import link_rows, normalize, simhash  # noqa: E402

BOARD_API = "https://projects.tas.twn.network/api/external/v1"
SPACE_ID = "sanguine-COpRXpkvCi19pKoDZnOPg0"
DEFAULT_PROJECT = "apeiron-ExNAtv3hObMmq01Ab20DfA"
SESSION_DB = os.path.expanduser("~/.local/share/opencode/opencode.db")
SANGUINE_REPOS = ("apeiron", "song", "intent", "sanguine", "fold", "resonance", "penglai")

# A staleness score is a weighted sum over named inputs. The weights are a
# design assumption, not a fit to data: they are stated here so a reader can
# replace them. Grade: `model`. The four inputs and their weights:
#   age_days        how long the row has sat In progress        0.40
#   no_named_commit no commit in --cycles naming the row         0.25
#   no_lane         no opencode session in the repo for the slug 0.20
#   quiet_source    no source named by the row moved since it    0.15
STALE_WEIGHTS = {"age_days": 0.40, "no_named_commit": 0.25, "no_lane": 0.20, "quiet_source": 0.15}
STALE_AGE_SCALE_DAYS = 14.0  # age_days/STALE_AGE_SCALE_DAYS saturates at 1.0

PATH_RE = re.compile(
    r"(?<![\w./-])((?:[\w.-]+/)+[\w.-]+\.(?:md|py|sh|lean|rs|toml|json|jsonl|lean|txt|lock|yml|yaml|sln|cfg))"
)
BARE_FILE_RE = re.compile(r"(?<![\w./-])([\w-]+\.(?:md|lean|rs|py))")
HEX64_RE = re.compile(r"\b([0-9a-f]{64})\b")


# ---------------------------------------------------------------- collection


def run(cmd, cwd, timeout=60, check=True):
    """Run a command, return (rc, stdout+stderr). Never raises on non-zero unless check."""
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {' '.join(cmd)}"
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def fetch_board(api_key, project_id):
    """Read the board over the verified REST route. No key is a named refusal."""
    import urllib.request

    if not api_key:
        return None, "OPERATELY_API_KEY absent; board classes A,B,C,H are not-established"
    url = f"{BOARD_API}/tasks/list?project_id={project_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            payload = json.load(r)
    except Exception as exc:  # noqa: BLE001 -- the failure is the datum here
        return None, f"board read failed: {type(exc).__name__}: {exc}"
    return payload.get("tasks", []), None


def api_key_from(repo):
    env = os.environ.get("OPERATELY_API_KEY")
    if env:
        return env
    path = os.path.join(repo, ".env")
    if os.path.exists(path):
        for line in open(path):
            if line.startswith("OPERATELY_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return None


def flat_text(description):
    """Operately descriptions are ProseMirror JSON; flatten to words.

    The corpus text is `content[].content[].text` and `content[].text`; a
    non-JSON description is used verbatim. Unparsed structure is dropped, not
    guessed at.
    """
    if not description:
        return ""
    if isinstance(description, str) and description.lstrip().startswith("{"):
        try:
            doc = json.loads(description)
        except json.JSONDecodeError:
            return description
        parts = []

        def walk(node):
            if isinstance(node, dict):
                if isinstance(node.get("text"), str):
                    parts.append(node["text"])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(doc)
        return " ".join(parts)
    return str(description)


def lane_index(db_path=SESSION_DB, window_hours=12.0):
    """directory -> newest session update, and the set of recent session titles.

    Read-only, no WAL write. A missing db is a named refusal for the lane term
    of class A, not a silent zero.
    """
    if not os.path.exists(db_path):
        return {}, {}, f"session db absent at {db_path}; lane term of class A is not-established"
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        cutoff = int((time.time() - window_hours * 3600) * 1000)
        rows = con.execute(
            "select coalesce(directory,''), coalesce(title,''), time_updated from session_v2 "
            "where time_updated > ? and directory is not null",
            (cutoff,),
        ).fetchall()
        con.close()
    except sqlite3.Error as exc:
        return {}, {}, f"session db read failed: {exc}; lane term of class A is not-established"
    by_dir, titles = {}, set()
    for d, t, ts in rows:
        by_dir[d] = max(by_dir.get(d, 0), ts)
        if t:
            titles.add(t)
    return by_dir, titles, None


# ------------------------------------------------------------------- class A


def row_slug(name):
    """`APEIRON-VV-V1-CHAIN: re-diagnose` -> `vv-v1-chain`.

    Two segments after the project prefix, lowercased. Two segments is what the
    lane titles actually carry (`VV-V1 re-diagnose ReadLoop`).
    """
    head = normalize(name.split(":")[0])
    segs = [s for s in head.split() if s]
    if not segs:
        return ""
    if len(segs) == 1:
        return segs[0]
    return f"{segs[0]} {segs[1]}" if segs[0] in ("m1", "m2", "m3") else " ".join(segs[1:3])


def repo_of(row):
    """Which repository a row belongs to, from its id prefix or its description."""
    blob = f"{row.get('id','')} {flat_text(row.get('description'))}".lower()
    for r in ("apeiron", "song", "intent"):
        if blob.startswith(r) or f"[{r}]" in blob or f"apeiron-" in row.get("id", "") and r == "apeiron":
            return r
    for r in SANGUINE_REPOS:
        if f"{r}/" in blob or f"{r}:" in blob:
            return r
    return None


def class_a(rows, repo_root, lane_dirs, lane_titles, cycles, now):
    rc, log = run(
        ["git", "log", f"-{cycles}", "--pretty=format:%H%x1f%at%x1f%B%x1e"],
        repo_root, timeout=120,
    )
    commits = []
    if rc == 0:
        for rec in log.split("\x1e"):
            rec = rec.strip("\n")
            if not rec.strip():
                continue
            h, ts, body = (rec.split("\x1f", 2) + ["", ""])[:3]
            commits.append({"sha": h, "ts": int(ts), "body": body})
    commit_text = "\n".join(c["body"].lower() for c in commits)
    out = []
    for r in rows:
        st = (r.get("status") or {}).get("value")
        if st != "in_progress":
            continue
        upd = r.get("updated_at") or ""
        age_days = 0.0
        if upd:
            try:
                age_days = (now - time.mktime(time.strptime(upd[:19], "%Y-%m-%dT%H:%M:%S"))) / 86400.0
            except ValueError:
                age_days = 0.0
        rid = r.get("id", "")
        slug = row_slug(r.get("name", ""))
        named_commit = bool(rid) and rid.lower() in commit_text
        if not named_commit and slug:
            named_commit = slug.replace(" ", "-") in commit_text.replace(" ", "-")
        repo = repo_of(r) or "apeiron"
        repo_dir = os.path.join(os.path.dirname(repo_root), repo)
        lane_hit = bool(slug) and any(slug in t.lower() for t in lane_titles)
        repo_lane = repo_dir in lane_dirs or any(
            d.startswith(repo_dir + os.sep) for d in lane_dirs
        )
        # sources named by the row, and their last commit time
        text = flat_text(r.get("description"))
        named_src = newest_source_commit(repo_dir, extract_paths(text))
        upd_ts = time.mktime(time.strptime(upd[:19], "%Y-%m-%dT%H:%M:%S")) if upd else 0
        quiet = 1.0 if (not named_src or named_src < upd_ts) else 0.0
        inputs = {
            "age_days": min(age_days / STALE_AGE_SCALE_DAYS, 1.0),
            "no_named_commit": 0.0 if named_commit else 1.0,
            "no_lane": 0.0 if (lane_hit or repo_lane) else 1.0,
            "quiet_source": quiet,
        }
        score = sum(STALE_WEIGHTS[k] * v for k, v in inputs.items())
        out.append(
            {
                "row": rid,
                "name": r.get("name", "")[:90],
                "repo": repo,
                "slug": slug,
                "age_days": round(age_days, 2),
                "inputs": {k: round(v, 2) for k, v in inputs.items()},
                "score": round(score, 3),
                "repo_has_lane": repo_lane,
                "lane_title_hit": lane_hit,
                "sources": named_src,
            }
        )
    out.sort(key=lambda x: -x["score"])
    return {
        "cycles_scanned": len(commits),
        "weights": STALE_WEIGHTS,
        "age_scale_days": STALE_AGE_SCALE_DAYS,
        "rows_with_a_named_commit": sum(1 for r in rows if r.get("in_progress") and r["inputs"]["no_named_commit"] == 0.0),
        "lane_window_hours": 12.0,
        "rows": out,
    }


# ------------------------------------------------------------------- class C


def extract_paths(text):
    found = set(PATH_RE.findall(text))
    found |= set(BARE_FILE_RE.findall(text))
    return sorted(p for p in found if len(p) > 3 and not p.startswith("."))


RESOLVE_ROOTS = ("", "proof/", "proof/corpus/", "tools/", "docs/", "src/", "tools/debt/")


def resolve_path(repo_root, path):
    for r in RESOLVE_ROOTS:
        cand = os.path.join(repo_root, r, path)
        if os.path.exists(cand):
            return os.path.relpath(cand, repo_root)
    return None


def class_c(rows, repo_root):
    out = []
    for r in rows:
        text = f"{r.get('name','')} {flat_text(r.get('description'))}"
        dangling = []
        for p in extract_paths(text):
            if resolve_path(repo_root, p) is None:
                dangling.append(p)
        if dangling:
            out.append({"row": r.get("id"), "name": r.get("name", "")[:80], "paths": dangling})
    return {"rows": out}


# ------------------------------------------------------------------- class D


def digest_map(repo_root):
    """Every (source, path, recorded_sha256) triple this corpus records, in two
    populations that must not be summed.

    `primary` -- the evidence ledger (`docs/evidence/ledger.jsonl`). Its `path`
    is documented as repo-relative, so an unresolvable path here is a genuine
    defect: a receipt that names a file which is gone.

    `secondary` -- any JSON/JSONL under artifacts/ and tools/ holding a map from
    a path to a 64-hex digest. These are *acceptance artifacts of whichever
    repository the acceptance ran against*, so their keys are frequently paths
    in a sibling repository (`proof/native/NarrowSource.lean` is not an apeiron
    path). Resolving them against this root and calling the failures "missing"
    is a category error: it reported 10,041 absent files on the live tree that
    are simply not this repository's files. They are reported as
    `unresolved_base` with the sibling repository they do resolve under, when one
    does.
    """
    primary, secondary = [], []
    ledger = os.path.join(repo_root, "docs/evidence/ledger.jsonl")
    if os.path.exists(ledger):
        for line in open(ledger):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("path") and rec.get("sha256"):
                primary.append({"src": "ledger.jsonl", "path": rec["path"],
                                "sha256": rec["sha256"], "id": rec.get("id")})
    for base in ("artifacts", "tools"):
        for root, dirs, files in os.walk(os.path.join(repo_root, base)):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules")]
            for fn in sorted(files):
                if not fn.endswith((".json", ".jsonl")):
                    continue
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, repo_root)
                try:
                    if fn.endswith(".json"):
                        obj = json.load(open(fp))
                    else:
                        obj = [json.loads(l) for l in open(fp) if l.strip()]
                except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                    continue
                secondary.extend(_walk_digests(obj, rel))
    return primary, secondary


def _walk_digests(node, src, acc=None):
    acc = [] if acc is None else acc
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and isinstance(v, str) and HEX64_RE.fullmatch(v):
                if "/" in k or "." in k:
                    acc.append({"src": src, "path": k, "sha256": v, "id": None})
            if isinstance(v, (dict, list)):
                _walk_digests(v, src, acc)
    elif isinstance(node, list):
        for v in node:
            _walk_digests(v, src, acc)
    return acc


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Secondary digest claims carry no recorded base. These are the bases observed on
# this tree, in the order tried: the repo root, the two lake package roots (the
# 9,341 `packages/**.lean` keys are Mathlib/Hypothesis identities whose base is
# `proof/corpus/.lake/`), then sibling repository roots.
SECONDARY_BASES = ("", "proof/corpus/.lake/", "proof/.lake/", "proof/corpus/")


def class_d(repo_root):
    primary, secondary = digest_map(repo_root)
    workspace = os.path.dirname(repo_root)
    siblings = sorted(
        d for d in (os.listdir(workspace) if os.path.isdir(workspace) else [])
        if os.path.isdir(os.path.join(workspace, d))
    )
    led_drift, led_missing, ok = [], [], 0
    for c in primary:
        p = os.path.join(repo_root, c["path"])
        if not os.path.exists(p):
            led_missing.append(c)
            continue
        actual = sha256_file(p)
        if actual != c["sha256"]:
            led_drift.append({**c, "actual": actual})
        else:
            ok += 1
    sec_drift, sec_ok, by_base, sec_missing = [], 0, {}, []
    for c in secondary:
        located = None
        if os.path.isabs(c["path"]):
            # An absolute key names a file outside the repo-root-relative
            # scheme; report which repository it points into rather than
            # crediting the base that happened to make os.path.join work.
            if os.path.exists(c["path"]):
                inside = c["path"].startswith(repo_root + os.sep)
                located = ("this repo (absolute)" if inside else "absolute, outside this repo", c["path"])
        else:
            for base in SECONDARY_BASES:
                full = os.path.join(repo_root, base, c["path"])
                if os.path.exists(full):
                    located = (f"this repo:{base or '.'}", full)
                    break
            if located is None:
                for sib in siblings:
                    full = os.path.join(workspace, sib, c["path"])
                    if os.path.isfile(full):
                        located = (f"sibling:{sib}", full)
                        break
        if located is None:
            sec_missing.append(c)
            continue
        label, full = located
        by_base[label] = by_base.get(label, 0) + 1
        actual = sha256_file(full)
        if actual != c["sha256"]:
            sec_drift.append({**c, "actual": actual, "base": label})
        else:
            sec_ok += 1
    return {
        "primary": {
            "source": "docs/evidence/ledger.jsonl",
            "claims": len(primary), "ok": ok,
            "drifted": led_drift, "missing_path": led_missing,
        },
        "secondary": {
            "note": "acceptance artifacts; a claim's base is not recorded, so it is resolved against a declared base set",
            "bases_tried": list(SECONDARY_BASES) + [f"sibling:<repo>" for _ in siblings],
            "claims": len(secondary), "ok": sec_ok,
            "located_by_base": by_base,
            "drifted": sec_drift,
            "unresolved_anywhere": sec_missing,
        },
    }


# ------------------------------------------------------------------- class E


def newest_source_commit(repo_root, paths, cache=None):
    """Newest commit unix time among `paths` that are tracked in `repo_root`."""
    best = 0
    cache = cache if cache is not None else {}
    for p in paths:
        full = os.path.join(repo_root, p)
        if not os.path.exists(full) or not os.path.isfile(full):
            continue
        if p not in cache:
            rc, out = run(["git", "log", "-1", "--format=%ct", "--", p], repo_root, timeout=30)
            cache[p] = int(out.strip()) if rc == 0 and out.strip().isdigit() else 0
        best = max(best, cache[p])
    return best or None


GATE_RE = re.compile(r"\b(cargo \w+|lake build|proof/regen\.sh|proof/model-patches\.sh|python3 [\w./-]+\.py)\b")


def class_e(repo_root, commits_index):
    """A receipt whose own commit predates a source it names: the gate it records
    cannot have run after that source moved. mtime is not used -- git order is."""
    receipts = []
    for root, dirs, files in os.walk(os.path.join(repo_root, "docs/evidence")):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in sorted(files):
            if fn.endswith(".md"):
                receipts.append(os.path.relpath(os.path.join(root, fn), repo_root))
    rc, out = run(["git", "ls-files", "docs/evidence"], repo_root, timeout=60)
    tracked = set(out.split()) if rc == 0 else set()
    cache, stale = {}, []
    for rel in receipts:
        rc_t, ts = run(["git", "log", "-1", "--format=%ct", "--", rel], repo_root, timeout=30)
        if rc_t != 0 or not ts.strip().isdigit():
            continue  # untracked receipt: no git order to compare, named in class F
        receipt_ts = int(ts.strip())
        body = open(os.path.join(repo_root, rel), errors="replace").read()
        gates = sorted(set(GATE_RE.findall(body)))
        named = [p for p in extract_paths(body) if resolve_path(repo_root, p)]
        src_ts = newest_source_commit(repo_root, named, cache)
        if src_ts and src_ts > receipt_ts:
            stale.append(
                {
                    "receipt": rel,
                    "gates": gates,
                    "newer_source": src_ts,
                    "receipt_commit": receipt_ts,
                    "lag_days": round((src_ts - receipt_ts) / 86400.0, 2),
                    "tracked": rel in tracked,
                }
            )
    stale.sort(key=lambda r: -r["lag_days"])
    return {"receipts_scanned": len(receipts), "stale": stale}


# ------------------------------------------------------------------- class F


def class_f(repo_root):
    """Receipt files against the index, in both directions.

    Every file under docs/evidence counts, not only `.md`: the index carries rows
    for `.json`, `.rs` and `.patch` artifacts too, and a `.md`-only walk reports
    those four live files as dangling index rows.
    """
    ledger = os.path.join(repo_root, "docs/evidence/ledger.jsonl")
    files = set()
    for root, dirs, fns in os.walk(os.path.join(repo_root, "docs/evidence")):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in fns:
            rel = os.path.relpath(os.path.join(root, fn), repo_root)
            if rel != "docs/evidence/ledger.jsonl":
                files.add(rel)
    rows = {}
    if os.path.exists(ledger):
        for line in open(ledger):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("path"):
                rows[rec["path"]] = rec.get("id")
    unindexed = sorted(files - set(rows))
    return {
        "files": len(files),
        "markdown_receipts": sum(1 for f in files if f.endswith(".md")),
        "rows": len(rows),
        "unindexed": unindexed,
        "dangling": [{"path": p, "id": i} for p, i in sorted(rows.items()) if p not in files],
    }


# ------------------------------------------------------------------- class G


TASK_ID_RE = re.compile(r"\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+){1,7})\b")


def class_g(repo_root, rows):
    rc, wt_out = run(["git", "worktree", "list", "--porcelain"], repo_root, timeout=60)
    worktrees = []
    cur = {}
    for line in wt_out.splitlines() + [""]:
        if not line:
            if cur:
                worktrees.append(cur)
                cur = {}
            continue
        k, _, v = line.partition(" ")
        cur[k] = v
    rc_b, br_out = run(["git", "branch", "--format=%(refname:short)\t%(objectname:short)\t%(upstream:short)\t%(worktreepath)"], repo_root, timeout=60)
    branches = []
    for line in br_out.splitlines():
        parts = line.split("\t")
        if parts and parts[0]:
            branches.append(
                {"name": parts[0], "sha": parts[1] if len(parts) > 1 else "",
                 "upstream": parts[2] if len(parts) > 2 else "",
                 "worktree": parts[3] if len(parts) > 3 else ""}
            )
    ids = [r.get("id", "") for r in rows]
    names = " \n".join(f"{r.get('id','')} {r.get('name','')}" for r in rows)
    head = run(["git", "rev-parse", "--short", "HEAD"], repo_root, timeout=30)[1].strip()
    main_ts = newest_source_commit(repo_root, [])  # unused placeholder guard
    rc_log, log = run(["git", "log", "-1", "--format=%ct"], repo_root, timeout=30)
    main_ct = int(log.strip()) if rc_log == 0 and log.strip().isdigit() else 0

    wt_rows = []
    for w in worktrees:
        path = w.get("worktree", "")
        if path == repo_root:
            continue
        sha = w.get("HEAD", "")
        detached = "detached" in w or not w.get("branch")
        named = None
        for rid, nm in zip(ids, [r.get("name", "") for r in rows]):
            if rid and rid in path:
                named = rid
                break
        if not named:
            for m in TASK_ID_RE.findall(os.path.basename(path)):
                if m in names:
                    named = m
                    break
        wt_rows.append(
            {
                "path": path,
                "head": sha[:8],
                "at_main": sha.startswith(head),
                "detached": detached,
                "row": named,
            }
        )
    wt_rows.sort(key=lambda w: (w["row"] is not None, w["at_main"]))
    br_rows = []
    for b in branches:
        named = next((rid for rid in ids if rid and rid in b["name"]), None) or (
            b["name"] if b["name"] in names else None
        )
        br_rows.append({**b, "row": named})
    # rows that name a worktree or branch
    anchored = {w["row"] for w in wt_rows if w["row"]} | {b["row"] for b in br_rows if b["row"]}
    unanchored = [
        {"row": r.get("id"), "name": r.get("name", "")[:70]}
        for r in rows
        if r.get("id") and r["id"] not in anchored and r.get("status", {}).get("value") == "in_progress"
    ]
    return {
        "worktrees_total": len(wt_rows),
        "worktrees_detached": sum(1 for w in wt_rows if w["detached"]),
        "worktrees_not_at_main": sum(1 for w in wt_rows if not w["at_main"]),
        "worktrees_without_row": [w for w in wt_rows if not w["row"]],
        "branches": br_rows,
        "branches_without_row": [b for b in br_rows if not b["row"]],
        "in_progress_rows_without_git_state": unanchored,
    }


# ------------------------------------------------------------------- class H


BOARD_LINE_RE = re.compile(r"^\s*board:\s*(.+?)\s*$", re.MULTILINE)


def class_h(repo_root, rows, ids_set, commits=80):
    """Commit-side board declaration.

    The `board:` line's presence reads from git alone, so it is reported even
    when the board is unreachable; only the "declared row still exists" arm needs
    the board, and it is `not_established` without one rather than a false pass.
    """
    rc, log = run(["git", "log", f"-{commits}", "--pretty=format:%H%x1f%B%x1e"], repo_root, timeout=120)
    out, missing, dangling, named_rows = [], [], [], set()
    if rc == 0:
        for rec in log.split("\x1e"):
            rec = rec.strip("\n")
            if not rec.strip():
                continue
            parts = rec.split("\x1f", 1)
            sha, body = parts[0], parts[1] if len(parts) > 1 else ""
            m = BOARD_LINE_RE.search(body)
            if not m:
                missing.append({"sha": sha[:8], "subject": body.strip().splitlines()[0][:80] if body.strip() else ""})
                continue
            ref = m.group(1).strip()
            if ref.startswith("none"):
                continue  # housekeeping / spike: declared absence, not a missing row
            named_rows.add(ref)
            if ids_set is not None and ref not in ids_set:
                dangling.append({"sha": sha[:8], "declared": ref})
    result = {
        "commits_scanned": commits,
        "commits_without_declaration": missing,
        "distinct_rows_declared": len(named_rows),
    }
    if ids_set is None:
        result["declarations_naming_absent_row"] = "not-established (no board read)"
    else:
        result["declarations_naming_absent_row"] = dangling
    return result


# ---------------------------------------------------------------------- main


TRUST_BOUNDARY = """TRUST BOUNDARY
  Cannot see: task comments (Operately exposes comments_count only on the list
  route); rows private to another project; semantic debt that is not one of the
  eight named signals; any lane on another box; git objects not reachable from
  this repository's refs.
  Class D trusts the recorded digest's spelling of its own path: a record naming
  the wrong path is not detected, only a record whose named file changed.
  Class E orders by git commit time, so a receipt committed in the same second
  as its source is not flagged; it does not observe whether a gate ran.
  Class A's lane term reads one session table on this box at one instant.
  Class B's match distribution is estimated from an exact-name seed, which is a
  matched-by-construction sample: the scores are an index for a human, and no
  pair is a proved duplicate.
  No mutation of the live tree occurs: this program opens every file read-only
  and the board route is a GET."""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--board", default=None,
                    help="read a board capture from a JSON file instead of REST; "
                         "the shape is the REST tasks/list payload. Used by the controls.")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--cycles", type=int, default=120, help="commits scanned for class A/H")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    now = time.time()

    if args.board:
        rows = json.load(open(args.board)).get("tasks", [])
        board_err = None
        board_source = f"capture:{args.board}"
    else:
        key = api_key_from(repo)
        rows, board_err = fetch_board(key, args.project)
        board_source = "rest" if rows is not None else "unavailable"
    lane_dirs, lane_titles, lane_err = lane_index()

    result = {
        "repo": repo,
        "project": args.project,
        "board_source": board_source,
        "read_at_unix": now,
        "board_error": board_err,
        "lane_error": lane_err,
        "board_rows": len(rows) if rows else 0,
        "lane_window_hours": 12.0,
    }
    # The repository classes do not read the board, so they run on every path: a
    # board outage must not silence digest drift or an orphaned receipt.
    result["classes"] = {
        "D": class_d(repo),
        "E": class_e(repo, None),
        "F": class_f(repo),
        "H": class_h(repo, rows, None if rows is None else {r.get("id") for r in rows}, args.cycles),
    }
    if rows is None:
        for k in ("A", "B", "C", "G"):
            result["classes"][k] = {"not_established": board_err}
    else:
        names = {r["id"]: r.get("name", "") for r in rows if r.get("id")}
        descs = {r["id"]: flat_text(r.get("description")) for r in rows if r.get("id")}
        result["classes"].update(
            {
                "A": class_a(rows, repo, lane_dirs, lane_titles, args.cycles, now),
                "B": {"link": link_rows(names, descs)},
                "C": class_c(rows, repo),
                "G": class_g(repo, rows),
            }
        )

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    print_report(result, args.top)
    return 0


def print_report(r, top):
    L = print
    L("=" * 78)
    L("PLATFORM DEBT DETECTOR  --  APEIRON-PLATFORM-DEBT-DETECTOR")
    L(f"repo={r['repo']}")
    L(f"project={r['project']}  board_source={r.get('board_source')}  rows={r['board_rows']}  read_at={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(r['read_at_unix']))}")
    if r["board_error"]:
        L(f"BOARD REFUSAL: {r['board_error']}")
    if r["lane_error"]:
        L(f"LANE REFUSAL: {r['lane_error']}")
    L("=" * 78)
    C = r.get("classes", {})

    a = C.get("A", {})
    if "rows" in a:
        L("")
        L(f"[A] STALE IN-PROGRESS  rows_in_progress={len(a['rows'])} of {r['board_rows']}  commits_scanned={a['cycles_scanned']}  lane_window={a['lane_window_hours']}h")
        L(f"    weights (model, uncalibrated) {a['weights']}  age saturates at {a['age_scale_days']}d")
        L(f"    in-progress rows with a commit naming them: {a['rows_with_a_named_commit']} of {len(a['rows'])}")
        for x in a["rows"][:top]:
            L(f"    {x['score']:.3f}  {x['row'][:46]:46} age={x['age_days']:5.2f}d repo={x['repo']:10} inputs={x['inputs']}")
    else:
        L(f"\n[A] not-established: {a.get('not_established')}")

    b = C.get("B", {}).get("link")
    if b:
        L("")
        L(f"[B] DUPLICATE ROWS  mode={b['mode']}  reported={b['reported']} of {b['candidates']} candidates")
        L(f"    rule: {b['decision_rule']}; declared prior={b['prior_duplicate_rate']} (model)")
        L(f"    m seed: {b['m_seed_kind']}")
        L(f"    u={b['u']}")
        L(f"    m={b['m']}")
        L(f"    identified={b['identified']} not_identified={b['not_identified']} (excluded, not clamped)")
        for p in b["pairs"][:top]:
            L(f"    post={p['posterior']:.4f} flip_at_prior={p['break_even_prior']:.4f} llr={p['score']:7.2f}b ham={p['hamming']:2} {p['a'][:34]:34} <-> {p['b'][:34]}")

    c = C.get("C", {})
    if "rows" in c:
        L("")
        L(f"[C] DANGLING REFERENCE  rows_naming_unresolvable_path={len(c['rows'])}")
        for x in c["rows"][:top]:
            L(f"    {x['row'][:46]:46} {','.join(x['paths'])[:70]}")

    d = C.get("D", {})
    if "primary" in d:
        p, s = d["primary"], d["secondary"]
        L("")
        L(f"[D] DIGEST DRIFT -- primary (evidence ledger, repo-relative): claims={p['claims']} ok={p['ok']} drifted={len(p['drifted'])} missing_path={len(p['missing_path'])}")
        for x in p["drifted"][:top]:
            L(f"    DRIFT   {x['path'][:56]:56} recorded={x['sha256'][:16]} actual={x['actual'][:16]}")
        for x in p["missing_path"][:top]:
            L(f"    MISSING {x['path'][:56]:56} id={x['id']}")
        L(f"[D] secondary (acceptance artifacts, base not recorded): claims={s['claims']} ok={s['ok']} drifted={len(s['drifted'])} unresolved_anywhere={len(s['unresolved_anywhere'])}")
        L(f"    located by base: {s['located_by_base']}")
        for x in s["drifted"][:top]:
            L(f"    DRIFT   {x['path'][:52]:52} base={x['base'][:18]:18} src={x['src'][:26]}")
            L(f"            recorded={x['sha256'][:16]} actual={x['actual'][:16]}")

    e = C.get("E", {})
    if "stale" in e:
        L("")
        L(f"[E] UNREFRESHED EVIDENCE  receipts={e['receipts_scanned']} receipt_predates_a_source_it_names={len(e['stale'])}")
        for x in e["stale"][:top]:
            L(f"    lag={x['lag_days']:7.2f}d  {x['receipt'][:58]:58} gates={','.join(x['gates'])[:40]}")

    f = C.get("F", {})
    if "files" in f:
        L("")
        L(f"[F] ORPHANED RECEIPT  files={f['files']} (markdown_receipts={f['markdown_receipts']}) index_rows={f['rows']} unindexed={len(f['unindexed'])} dangling_rows={len(f['dangling'])}")
        for p in f["unindexed"][:top]:
            L(f"    UNINDEXED {p}")
        for x in f["dangling"][:top]:
            L(f"    DANGLING  {x['path']}  id={x['id']}")

    g = C.get("G", {})
    if "worktrees_total" in g:
        L("")
        L(f"[G] UNANCHORED GIT STATE  worktrees={g['worktrees_total']} detached={g['worktrees_detached']} not_at_main={g['worktrees_not_at_main']} branches={len(g['branches'])}")
        L(f"    worktrees naming no board row: {len(g['worktrees_without_row'])}")
        for w in g["worktrees_without_row"][:top]:
            L(f"    {w['head']} detached={str(w['detached']):5} at_main={str(w['at_main']):5} {w['path']}")
        L(f"    branches naming no board row: {len(g['branches_without_row'])}")
        for b in g["branches_without_row"][:top]:
            L(f"    {b['name'][:40]:40} {b['sha']} -> {b['upstream'] or '-'} wt={b['worktree'] or '-'}")
        L(f"    in-progress rows with no worktree/branch: {len(g['in_progress_rows_without_git_state'])}")
        for x in g["in_progress_rows_without_git_state"][:top]:
            L(f"    {x['row'][:46]:46} {x['name']}")

    h = C.get("H", {})
    if "commits_without_declaration" in h:
        L("")
        absent = h["declarations_naming_absent_row"]
        L(f"[H] UNDECLARED COMMIT  commits={h['commits_scanned']} without_a_board_line={len(h['commits_without_declaration'])} distinct_rows_declared={h['distinct_rows_declared']} naming_an_absent_row={absent if isinstance(absent, str) else len(absent)}")
        for x in h["commits_without_declaration"][:top]:
            L(f"    {x['sha']}  {x['subject']}")
        if isinstance(absent, list):
            for x in absent[:top]:
                L(f"    ABSENT ROW {x['sha']} declares {x['declared']}")

    L("")
    L(TRUST_BOUNDARY)
    L("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())