#!/usr/bin/env python3
# Row: `apeiron-checkpoint-re-the-FVDhqA987JrTcG3bUGFRP9` (APEIRON-CHECKPOINT-CONTROLS).
# Row: `apeiron-docs-repoint-Hhlc7OjTttFQLPuENEoekh` (APEIRON-DOCS-REPOINT).
"""The per-cycle board checkpoint: measure -> check-in -> row moves -> marker.

``AGENTS.md`` §3 step 5 makes the board update a field of a checkpoint lane's own
return, and the §3 dispatch loop names two enforcers for it. Neither was one: the
per-commit hook required a board *row declaration* (``commit-msg``), never a
check-in, and the only trigger was the operator pulse -- a timer that can be
switched off, with no failure when the step is skipped. This tool anchors the
obligation to an event that already happens, **a commit that changes the source
the number is measured on**, and makes the board update a byproduct of the tool
that produced the number.

The mechanism
=============

One command, three outputs:

1. **measure** -- read a measurement acceptance stream (the ``key=value`` lines a
   comparison harness prints; the ``examples/reg1`` harness was removed in
   ``c9a2d0c``) and reduce it to the headline: each
   ``hashtrinity_*`` incumbent's ratio mean over its decided pairs, the worst
   cell, the digest fails, and the run's own ``acceptance_summary`` tally.
2. **check-in and rows** -- ``--dry-run`` prints the exact check-in text and the
   row moves and writes a marker holding the **digest of the check-in text**;
   ``--post`` sends that text to the Operately REST endpoint
   ``POST {base}/projects/create_check_in``, reads it back from
   ``GET {base}/projects/list_check_ins``, moves each named row with
   ``POST {base}/tasks/update_status`` and reads each one back, and writes a
   marker holding the check-in id. The REST transport is ``tools/board/board.py``'s
   own ``RestClient``: the credential never leaves that object, and the read-back
   is the verdict because this instance answers ``500`` for a mutation that
   landed.
3. **marker** -- ``.apeiron/checkpoint.json`` (override with ``--marker``): the
   check-in id or the dry-run digest, the sha256 of the measurement file, the
   **source digest of the staged and the working tree** over the source prefixes
   (``src/``; ``examples/reg1/`` was removed in ``c9a2d0c``), the row moves with
   their read-backs, and the time.

``checkpoint.py check`` then asks one question of the staged tree: does the marker's
staged digest equal the digest of the tree a commit would create? It refuses when
a path under ``src/`` (formerly also ``examples/reg1/``) is staged against a
marker that does not cover it, so a checkpoint that was skipped fails the query.

No hook runs that query. The per-cycle checkpoint step of
``.githooks/pre-commit`` was retired with the acceptance harness in ``c9a2d0c``
(owner direction 2026-10-05): it read a reg1 acceptance stream, and with
``examples/reg1`` removed there is none. The query is a standalone command a
checkpoint lane runs against its own staged tree, and the wave barrier remains
the authority on a commit that skipped it.

Trust boundary
==============

* The hook enforces only on the commit path (``git commit --no-verify`` skips it).
* ``--post`` writes only through the endpoint named above, with the credential in
  ``$OPERATELY_API_KEY`` (or ``.env`` in the repository root, untracked).
* ``--dry-run`` is the offline path and needs no credential: the marker then
  carries the digest of the check-in text instead of a check-in id, and the hook
  reports which of the two the marker holds.
* A digest binds a commit to a marker; it does not assert that the measurement
  behind the marker was honest. That is the measurement's own gate.

Usage
=====

::

    python3 tools/board/checkpoint.py measure --measure PATH
    python3 tools/board/checkpoint.py dry-run --measure PATH --row ID=STATUS
    python3 tools/board/checkpoint.py post --measure PATH --row ID=STATUS
    python3 tools/board/checkpoint.py check          # the staged-tree query; no hook runs it
    python3 tools/board/checkpoint.py self-test
    python3 -m unittest discover -s tools/board -p 'test_checkpoint.py'

``--measure PATH`` names the acceptance stream. Without it the newest ``*.json``
under the stated defaults is used: ``$APEIRON_REG1_ROOT`` if set, else
``/home/socol/oc-tmp/reg1-remeasure`` -- the reg1 harness writes its run streams
outside the repository (AGENTS.md §10, the tmpfs row), so there is no in-tree
default to guess. ``--measure-root`` adds a root; the newest stream by mtime
wins and its path and sha256 are recorded in the marker.

Exit status: ``0`` clean or skipped; ``1`` a finding (a missing, stale or
unlanded checkpoint); ``2`` usage, transport or a control-suite failure.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import board  # noqa: E402 -- the sibling driver; its REST transport is reused verbatim
except ImportError as exc:  # pragma: no cover - reported, never swallowed
    board = None  # type: ignore[assignment]
    BOARD_IMPORT_ERROR = str(exc)
else:
    BOARD_IMPORT_ERROR = ""

SCHEMA = 1
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The prefixes whose content the acceptance numbers are measured on: a commit
# that changes one changes the number, so a commit that changes one is what the
# checkpoint obligation is anchored to. `examples/reg1/` was removed in
# `c9a2d0c` and is no longer a source prefix.
SOURCE_PREFIXES = ("src/",)
# The three hashtrinity incumbents the programme reports. `hashtrinity_table` is
# the load-bearing serial table; `hashtrinity_count_par` and
# `hashtrinity_get_each` are the two parallel faces.
HEADLINE_INCUMBENTS = ("hashtrinity_table", "hashtrinity_count_par", "hashtrinity_get_each")
DEFAULT_MEASURE_ROOTS = ("/home/socol/oc-tmp/reg1-remeasure",)
DEFAULT_MARKER = ".apeiron/checkpoint.json"
DEFAULT_PROJECT_ID = "apeiron-ExNAtv3hObMmq01Ab20DfA"
DEFAULT_BASE_URL = board.DEFAULT_BASE_URL if board is not None else ""
CHECK_IN_ENDPOINT = "projects/create_check_in"
LIST_CHECK_INS_ENDPOINT = "projects/list_check_ins"
UPDATE_STATUS_ENDPOINT = "tasks/update_status"
TASK_ENDPOINT = "tasks/get"
# The enum the instance accepts for a check-in status, measured: a POST with
# `status=red` answers HTTP 400
# `Invalid value for enum project_check_in_status: red. Allowed values:
# on_track, caution, off_track`. `board.py`'s own `red`/`green` vocabulary is
# therefore not this endpoint's, so the default is derived from the measurement
# rather than copied from the driver.
CHECK_IN_STATUSES = ("on_track", "caution", "off_track")
HASH_TIMEOUT_S = 60

# A `key=value` pair. The value is a double-quoted string (which holds spaces
# and semicolons: the harness prints `reason="..."` and `rule="..."`), a
# bracketed list (which holds spaces: `pass_at_campaign=[64, 62, 62]`), or a run
# of non-space characters.
KV = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|\[[^\]]*\]|[^\s]*)')
# The record name a harness line opens with: `acceptance_cell_ratio cell=... ...`.
# It is a bare identifier, so a `key=value` line (`apeiron_head=...`) does not
# match -- that line's name *is* the key, and no record name follows it.
RECORD = re.compile(r'^([a-z][a-z0-9_]*)(?=[\s=]|$)')
# The keys this tool reads. Every other line of the stream is left alone, so an
# unread key is not a parse failure -- it is simply not part of the headline.
KEY_SUMMARY = "acceptance_summary"
KEY_REPEATS_HEADER = "acceptance_repeats"
KEY_INCUMBENT_RATIO = "acceptance_ratio"
KEY_INCUMBENT_MEAN = "acceptance_repeats_incumbent"
KEY_CELL_RATIO = "acceptance_cell_ratio"
KEY_ADDRESS_FACE = "acceptance_address_face"

# Finding rule names. Stable interface: the controls assert on them by name, so a
# rename breaks a control instead of silently weakening it.
RULE_NO_MARKER = "checkpoint-marker-missing"
RULE_STALE_MARKER = "checkpoint-marker-stale"
RULE_WRITE_NOT_LANDED = "checkpoint-check-in-not-landed"
RULE_ROW_NOT_LANDED = "checkpoint-row-not-landed"
RULE_MEASURE_UNREADABLE = "checkpoint-measure-unreadable"


class CheckpointError(RuntimeError):
    """A refusal that names its own rule, so the diagnostic is attributable."""


def finding(rule: str, detail: str, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"rule": rule, "detail": detail}
    item.update(extra)
    return item


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The acceptance stream
# ---------------------------------------------------------------------------


def parse_kv(line: str) -> dict[str, str]:
    """The ``key=value`` fields of one harness line, quotes honoured.

    ``line.split()`` is wrong for this stream: ``reason="a placement-only
    TableAddress is not the production address"`` is one field, not five. A
    whitespace split corrupts exactly the values a reader needs to quote back
    (``tools/board/checkpoint.py`` control ``kv-quotes-survive`` seeds a line
    whose quoted value holds spaces and asserts they do).
    """
    out: dict[str, str] = {}
    for key, value in KV.findall(line):
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        out[key] = value
    return out


def _as_float(fields: dict[str, str], key: str) -> float | None:
    raw = fields.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _as_int(fields: dict[str, str], key: str) -> int | None:
    raw = fields.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def read_measure(path: str) -> dict[str, Any]:
    """Reduce a ``reg1`` acceptance stream to the headline the check-in carries.

    Temporal cost: one pass over the stream, O(lines x fields per line) with
    ``fields`` bounded by the line's own width; a 2208-line full-profile run is
    read in well under a second, and the per-incumbent worst cell is selected in
    the same pass (no second scan, no sort).
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    summary: dict[str, str] = {}
    means: dict[str, dict[str, Any]] = {}
    ratios: dict[str, dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    faces: dict[str, dict[str, int]] = {}
    repeats_header: dict[str, str] = {}
    for line in text.splitlines():
        fields = parse_kv(line)
        opened = RECORD.match(line)
        key = opened.group(1) if opened else ""
        if key == KEY_SUMMARY:
            summary = fields
        elif key == KEY_REPEATS_HEADER and "axis" in fields:
            repeats_header = fields
        elif key == KEY_INCUMBENT_MEAN:
            incumbent = fields.get("incumbent", "")
            means[incumbent] = {
                "decided_pairs": _as_int(fields, "decided_pairs"),
                "ratio_mean": _as_float(fields, "ratio_mean"),
                "ratio_min": _as_float(fields, "ratio_min"),
                "ratio_max": _as_float(fields, "ratio_max"),
                "ratio_spread": _as_float(fields, "ratio_spread"),
                "pass_at_campaign": fields.get("pass_at_campaign", ""),
            }
        elif key == KEY_INCUMBENT_RATIO:
            incumbent = fields.get("incumbent", "")
            ratios[incumbent] = {
                "pairs": _as_int(fields, "pairs"),
                "digest_fail": _as_int(fields, "digest_fail"),
                "class": fields.get("class", ""),
                "r_min": _as_float(fields, "r_min"),
                "r_max": _as_float(fields, "r_max"),
            }
        elif key == KEY_CELL_RATIO:
            cells.append(
                {
                    "cell": fields.get("cell", ""),
                    "incumbent": fields.get("incumbent", ""),
                    "campaigns": _as_int(fields, "campaigns"),
                    "ratio_mean": _as_float(fields, "ratio_mean"),
                    "ratio_min": _as_float(fields, "ratio_min"),
                    "ratio_max": _as_float(fields, "ratio_max"),
                    "ratio_spread_abs": _as_float(fields, "ratio_spread_abs"),
                    "pass_at_campaign": fields.get("pass_at_campaign", ""),
                    "pooled": fields.get("pooled", ""),
                }
            )
        elif key == KEY_ADDRESS_FACE:
            tally = faces.setdefault(fields.get("incumbent", ""), {"matched": 0, "unmatched": 0, "na": 0})
            face = fields.get("face", "")
            if face == "matched":
                tally["matched"] += 1
            elif face == "unmatched":
                tally["unmatched"] += 1
            else:
                tally["na"] += 1
    headline: dict[str, Any] = {}
    for incumbent in HEADLINE_INCUMBENTS:
        mean = means.get(incumbent)
        ratio = ratios.get(incumbent)
        if mean is None and ratio is None:
            continue
        worst = None
        for cell in cells:
            if cell["incumbent"] != incumbent or cell["ratio_max"] is None:
                continue
            if worst is None or float(cell["ratio_max"]) > float(worst["ratio_max"]):
                worst = cell
        headline[incumbent] = {
            "decided_pairs": (mean or {}).get("decided_pairs", (ratio or {}).get("pairs")),
            "ratio_mean": (mean or {}).get("ratio_mean"),
            "ratio_min": (mean or {}).get("ratio_min", (ratio or {}).get("r_min")),
            "ratio_max": (mean or {}).get("ratio_max", (ratio or {}).get("r_max")),
            "ratio_spread": (mean or {}).get("ratio_spread"),
            "pass_at_campaign": (mean or {}).get("pass_at_campaign"),
            "digest_fail": (ratio or {}).get("digest_fail"),
            "worst_cell": (worst or {}).get("cell"),
            "worst_cell_ratio_max": (worst or {}).get("ratio_max"),
            "worst_cell_pooled": (worst or {}).get("pooled"),
        }
    if not headline and not summary:
        raise CheckpointError(
            f"{RULE_MEASURE_UNREADABLE}: {path} carries no `{KEY_SUMMARY}`, "
            f"`{KEY_INCUMBENT_MEAN}` or `{KEY_INCUMBENT_RATIO}` line, so it is not a reg1 "
            "acceptance stream; name the run file with --measure"
        )
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
        "summary": summary,
        "summary_line": kv_line(summary),
        "repeats_header": repeats_header,
        "headline": headline,
        "address_faces": faces,
        "cells_read": len(cells),
    }


def kv_line(fields: dict[str, str]) -> str:
    """``key=value`` for every field, in insertion order; the stream's own spelling."""
    return " ".join(f"{key}={value}" for key, value in fields.items())


def newest_measure(roots: Iterable[str]) -> tuple[str, list[tuple[str, str]]]:
    """The newest *readable* acceptance stream under ``roots``, by mtime.

    Newest-first, not newest-anything: a run still being written is the newest
    file on disk and has no ``acceptance_summary`` line yet, so the newest-first
    scan tries it, rejects it by name and takes the newest that reads. The
    rejects are returned, so the choice is auditable rather than silent.

    Temporal cost: O(files under the roots x bytes of each file read), because a
    candidate is proved readable by reading it; the reg1 harness writes a handful
    of run streams, so the scan is bounded and reported.
    """
    candidates: list[tuple[float, str]] = []
    for root in roots:
        for path in glob.glob(os.path.join(os.path.expanduser(root), "*.json")):
            if os.path.isfile(path):
                candidates.append((os.path.getmtime(path), path))
    if not candidates:
        raise CheckpointError(
            f"{RULE_MEASURE_UNREADABLE}: no *.json acceptance stream under "
            f"{', '.join(roots)}; pass --measure PATH or set APEIRON_REG1_ROOT"
        )
    candidates.sort(reverse=True)
    skipped: list[tuple[str, str]] = []
    for _, path in candidates:
        try:
            read_measure(path)
        except (CheckpointError, OSError) as exc:
            skipped.append((path, str(exc)))
            continue
        return path, skipped
    detail = "; ".join(f"{path}: {reason}" for path, reason in skipped)
    raise CheckpointError(
        f"{RULE_MEASURE_UNREADABLE}: none of the {len(candidates)} *.json stream(s) under "
        f"{', '.join(roots)} reads as a reg1 acceptance stream -- {detail}"
    )


# ---------------------------------------------------------------------------
# Source digests: the binding between the number and the commit
# ---------------------------------------------------------------------------


def _git(repo_root: str, args: list[str], check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", repo_root, *args],
        capture_output=True,
        text=True,
        timeout=HASH_TIMEOUT_S,
    )
    if check and result.returncode != 0:
        raise CheckpointError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def _entries_digest(entries: list[tuple[str, str, str]]) -> str:
    """One sha256 over ``(mode, blob, path)`` triples, sorted by path.

    The blob id is git's own (``sha1`` of ``blob <len>\\0<content>``), so the index
    side (``git ls-files -s``) and the working-tree side (``git hash-object``)
    land in one digest space and are directly comparable. A path's mode is part
    of the line, so an executable-bit change is a source change.
    """
    digest = hashlib.sha256()
    for mode, blob, path in sorted(entries, key=lambda item: item[2]):
        digest.update(f"{mode} {blob} {path}\n".encode("utf-8"))
    return digest.hexdigest()


def staged_digest(repo_root: str) -> tuple[str, int, list[str]]:
    """The digest of the tree the next commit would contain, over the prefixes.

    Read from the index (``git ls-files -s``), so it is the committed content and
    not the working tree: an unstaged edit is not in the commit and must not
    stale a marker. A deletion is absent from the index, which is the honest
    reading. Temporal cost: O(index entries under the two prefixes) -- 127 on
    this tree, measured -- in one ``git`` invocation.
    """
    out = _git(repo_root, ["ls-files", "-s", "-z", "--", *SOURCE_PREFIXES])
    entries: list[tuple[str, str, str]] = []
    paths: list[str] = []
    for record in out.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        mode, blob, stage = meta.split()
        if stage != "0":
            raise CheckpointError(
                f"unmerged index entry under {' or '.join(SOURCE_PREFIXES)}: {path} "
                f"(stage {stage}); resolve the conflict before the checkpoint"
            )
        entries.append((mode, blob, path))
        paths.append(path)
    return _entries_digest(entries), len(entries), paths


def worktree_digest(repo_root: str) -> tuple[str, int, list[str]]:
    """The digest of the working tree under the prefixes, tracked and untracked.

    ``git ls-files -co --exclude-standard`` names every path git would stage
    under the prefixes (``-o`` picks up an untracked source file; excluded
    patterns are honoured), and ``git hash-object`` produces the same blob ids
    the index holds, so the two digests are comparable. Temporal cost: O(paths
    under the prefixes) -- one ``git hash-object`` for the whole set, 127 files
    on this tree, measured -- and O(bytes) for the read.

    A symlink under a prefix is hashed by its target's content here and by its
    link text in the index, so the two digests would differ while the commit is
    unchanged; none exists (``git ls-files -s -- src`` has no mode
    ``120000`` entry, measured).
    """
    listed = _git(
        repo_root, ["ls-files", "-co", "--exclude-standard", "-z", "--", *SOURCE_PREFIXES]
    )
    paths = [record for record in listed.split("\0") if record]
    existing = [path for path in paths if os.path.isfile(os.path.join(repo_root, path))]
    missing = [path for path in paths if path not in set(existing)]
    entries: list[tuple[str, str, str]] = []
    if existing:
        # `--stdin-paths` keeps a path with a space in it one path; the output
        # order matches the input order, so it is zip-able back onto the paths.
        hashed = subprocess.run(
            ["git", "-C", repo_root, "hash-object", "--stdin-paths"],
            input="\n".join(existing) + "\n",
            capture_output=True,
            text=True,
            timeout=HASH_TIMEOUT_S,
        )
        if hashed.returncode != 0:
            raise CheckpointError(f"git hash-object: {hashed.stderr.strip()}")
        blobs = hashed.stdout.split()
        if len(blobs) != len(existing):
            raise CheckpointError(
                f"git hash-object returned {len(blobs)} ids for {len(existing)} paths"
            )
        for path, blob in zip(existing, blobs):
            absolute = os.path.join(repo_root, path)
            mode = "100755" if os.access(absolute, os.X_OK) else "100644"
            entries.append((mode, blob, path))
    return _entries_digest(entries), len(existing), missing


# ---------------------------------------------------------------------------
# The check-in text
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def checkin_text(
    measure: dict[str, Any],
    project: str,
    staged: tuple[str, int],
    worktree: tuple[str, int],
    rows: list[tuple[str, str]],
) -> str:
    """The description the check-in carries. One stamp line, then the headline."""
    staged_digest_value, staged_count, _ = staged
    worktree_digest_value, worktree_count, _ = worktree
    lines = [
        f"apeiron-checkpoint project={project} "
        f"staged_digest={staged_digest_value[:16]} staged_files={staged_count} "
        f"source_digest={worktree_digest_value[:16]} source_files={worktree_count}",
        f"measure={measure['path']} sha256={measure['sha256']}",
        f"reg1 {measure['summary_line'] or 'no acceptance_summary line in this stream'}",
    ]
    header = measure["repeats_header"]
    if header:
        lines.append(f"reg1 repetitions {kv_line(header)}")
    for incumbent in HEADLINE_INCUMBENTS:
        entry = measure["headline"].get(incumbent)
        if not entry:
            continue
        lines.append(
            f"hashtrinity {incumbent} decided_pairs={_fmt(entry['decided_pairs'])} "
            f"ratio_mean={_fmt(entry['ratio_mean'])} ratio_min={_fmt(entry['ratio_min'])} "
            f"ratio_max={_fmt(entry['ratio_max'])} ratio_spread={_fmt(entry['ratio_spread'])} "
            f"digest_fail={_fmt(entry['digest_fail'])} "
            f"worst_cell={entry['worst_cell'] or '-'}@{_fmt(entry['worst_cell_ratio_max'])} "
            f"worst_pooled={entry['worst_cell_pooled'] or '-'} "
            f"pass_at_campaign={entry['pass_at_campaign'] or '-'}"
        )
    faces = measure["address_faces"]
    for incumbent in HEADLINE_INCUMBENTS:
        tally = faces.get(incumbent)
        if tally:
            lines.append(
                f"hashtrinity address_face {incumbent} matched={tally['matched']} "
                f"unmatched={tally['unmatched']} not_applicable={tally['na']}"
            )
    if rows:
        lines.append("rows " + " ".join(f"{row_id}:{target}" for row_id, target in rows))
    lines.append(
        "Gate: `cargo test --workspace --all-features --locked`; the acceptance number is the "
        "reg1 full profile, whose stream this check-in names."
    )
    lines.append("Checkpoint: `python3 tools/board/checkpoint.py check` (no hook runs it).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The marker
# ---------------------------------------------------------------------------


def marker_path(repo_root: str, override: str | None) -> str:
    return os.path.join(repo_root, override or DEFAULT_MARKER)


def read_marker(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CheckpointError(f"{path}: {exc}") from None
    except ValueError as exc:
        raise CheckpointError(f"{RULE_STALE_MARKER}: {path} is not JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise CheckpointError(f"{RULE_STALE_MARKER}: {path} is not a marker object")
    return payload


def write_marker(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def build_marker(
    mode: str,
    project: str,
    measure: dict[str, Any],
    staged: tuple[str, int],
    worktree: tuple[str, int],
    text: str,
    rows: list[dict[str, Any]],
    check_in_id: str,
) -> dict[str, Any]:
    staged_digest_value, staged_count, _ = staged
    worktree_digest_value, worktree_count, _ = worktree
    return {
        "schema": SCHEMA,
        "tool": "tools/board/checkpoint.py",
        "mode": mode,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": project,
        "check_in_digest": sha256_of(text),
        "check_in_id": check_in_id,
        "check_in_bytes": len(text),
        "measure_path": measure["path"],
        "measure_sha256": measure["sha256"],
        "measure_skipped_newer": [path for path, _ in measure.get("skipped_newer") or ()],
        "acceptance_summary": measure["summary_line"],
        "headline": measure["headline"],
        "source_prefixes": list(SOURCE_PREFIXES),
        "source_digest": worktree_digest_value,
        "source_files": worktree_count,
        "staged_digest": staged_digest_value,
        "staged_files": staged_count,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# The staged-tree query
# ---------------------------------------------------------------------------


def staged_paths(repo_root: str) -> list[str]:
    """Every staged path under the prefixes (``git diff --cached --name-only``).

    Used for the diagnostic only: the digest comes from the index, so a rename or
    a deletion is already inside it. Temporal cost: O(staged entries).
    """
    out = _git(
        repo_root,
        ["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRD", "--", *SOURCE_PREFIXES],
        check=False,
    )
    return [record for record in out.split("\0") if record]


def run_check(args: argparse.Namespace) -> int:
    """The staged-tree query. Refuse, with the fix named.

    Run as ``python3 -B tools/board/checkpoint.py check --repo-root ROOT``. No
    hook step runs it (retired with the acceptance harness in ``c9a2d0c``), so
    this command is the whole harness the rule is exercised under.
    """
    repo_root = os.path.abspath(args.repo_root)
    changed = staged_paths(repo_root)
    marker = marker_path(repo_root, args.marker)
    if not changed:
        print(
            f"checkpoint: SKIP no staged path under {' or '.join(SOURCE_PREFIXES)}; "
            "the commit changes no measured source"
        )
        return 0
    payload = read_marker(marker)
    if payload is None:
        item = finding(
            RULE_NO_MARKER,
            f"{len(changed)} staged path(s) under {' or '.join(SOURCE_PREFIXES)} "
            f"({', '.join(changed[:4])}{'…' if len(changed) > 4 else ''}) and no checkpoint "
            f"marker at {os.path.relpath(marker, repo_root)}; the board update is a required "
            "field of the return that measured this source",
            paths=changed,
        )
        print(f"checkpoint: FAIL {item['rule']}: {item['detail']}", file=sys.stderr)
        print(f"checkpoint: fix: python3 tools/board/checkpoint.py dry-run --measure PATH", file=sys.stderr)
        return 1
    current, count, _ = staged_digest(repo_root)
    recorded = payload.get("staged_digest", "")
    if payload.get("schema") != SCHEMA or not recorded or recorded != current:
        item = finding(
            RULE_STALE_MARKER,
            f"{os.path.relpath(marker, repo_root)} records staged_digest="
            f"{recorded[:16] or 'none'} ({payload.get('created', 'no time')}, mode="
            f"{payload.get('mode', 'unknown')}, check-in {payload.get('check_in_id') or payload.get('check_in_digest', 'none')[:16]}), "
            f"and the staged tree is {current[:16]} over {count} file(s) "
            f"({', '.join(changed[:4])}{'…' if len(changed) > 4 else ''}); the check-in does not "
            "cover the source this commit would create",
            recorded=recorded,
            staged=current,
        )
        print(f"checkpoint: FAIL {item['rule']}: {item['detail']}", file=sys.stderr)
        print(
            "checkpoint: fix: python3 tools/board/checkpoint.py dry-run --measure PATH "
            "(or post --row ID=STATUS to move the rows too), then commit",
            file=sys.stderr,
        )
        return 1
    identity = payload.get("check_in_id") or f"dry-run digest {payload.get('check_in_digest', '')[:16]}"
    print(
        f"checkpoint: OK staged_digest={current[:16]} files={count} "
        f"marker={os.path.relpath(marker, repo_root)} created={payload.get('created', '?')} "
        f"check-in {identity} measure={os.path.relpath(payload.get('measure_path', '?'), repo_root)} "
        f"sha256={str(payload.get('measure_sha256', ''))[:16]}"
    )
    worktree, worktree_count, missing = worktree_digest(repo_root)
    if worktree != payload.get("source_digest"):
        # Informational: the commit is bound to the marker, and an unstaged edit
        # is not in the commit. Reported because a number measured before that
        # edit does not describe the working tree.
        print(
            f"checkpoint: note the working tree under the prefixes is {worktree[:16]} over "
            f"{worktree_count} file(s), not the measured {str(payload.get('source_digest', ''))[:16]}"
            + (f"; {len(missing)} path(s) absent on disk" if missing else "")
            + " -- an unstaged change is not in this commit"
        )
    return 0


# ---------------------------------------------------------------------------
# The board write
# ---------------------------------------------------------------------------


def load_env(repo_root: str) -> None:
    """Put ``.env``'s ``OPERATELY_API_KEY`` in the environment when it is not set.

    The token is untracked and owner-provided (``AGENTS.md`` §9.3); reading the
    file keeps ``--post`` runnable from a shell that has not sourced it. The value
    is never printed: ``board.RestClient`` scrubs it from every message.
    """
    if os.environ.get(board.KEY_ENV):
        return
    env_path = os.path.join(repo_root, ".env")
    try:
        with open(env_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() == board.KEY_ENV:
                    os.environ[board.KEY_ENV] = value.strip().strip('"').strip("'")
                    return
    except OSError:
        return


def post_check_in(
    repo_root: str, project_id: str, base_url: str, status: str, text: str
) -> tuple[str, dict[str, Any]]:
    """Post the check-in and read it back. The read-back is the verdict.

    ``POST {base}/projects/create_check_in`` is the only write; the confirmation
    is ``GET {base}/projects/list_check_ins``, and the stamp checked is the first
    line of the text, which carries the two source digests -- so a check-in that
    cannot be told apart from another is not confirmed.
    """
    if board is None:
        raise CheckpointError(f"board.py is not importable ({BOARD_IMPORT_ERROR})")
    load_env(repo_root)
    client = board.RestClient(base_url=base_url)
    response: dict[str, Any] = {}
    # The write is the shared one: `RestClient.post_check_in` encodes the
    # description as rich content (a plain string answers HTTP 500 and does not
    # land on this instance). Its transport is not the verdict — the read-back
    # below is, and this instance 500s on a mutation that landed.
    transport, transport_error = client.post_check_in(project_id, status, text)
    stamp = text.splitlines()[0]
    confirmed = False
    read_ids: list[str] = []
    try:
        for entry in client.check_ins(project_id):
            content = board.flatten_description(entry.get("content") or entry.get("description"))
            read_ids.append(str(entry.get("id", "")))
            if stamp in content:
                confirmed = True
    except board.TransportError as exc:
        print(
            f"checkpoint: note the read-back GET {base_url}/{LIST_CHECK_INS_ENDPOINT} failed: {exc}",
            file=sys.stderr,
        )
    check_in_id = ""
    for candidate in (response.get("id"), (response.get("check_in") or {}).get("id")):
        if candidate:
            check_in_id = str(candidate)
            break
    if confirmed and not check_in_id:
        # The POST answered without an id (or with an error); the read-back is
        # what identifies the check-in, so the newest matching id is recorded.
        for entry_id in reversed(read_ids):
            check_in_id = entry_id
            break
    if not confirmed:
        print(
            f"checkpoint: FAIL {RULE_WRITE_NOT_LANDED}: POST {base_url}/{CHECK_IN_ENDPOINT} "
            f"answered transport={transport}"
            + (f" ({transport_error})" if transport_error else "")
            + f", but GET {base_url}/{LIST_CHECK_INS_ENDPOINT} returned no check-in carrying "
            "the stamp; no marker is written and the commit stays refused",
            file=sys.stderr,
        )
    return check_in_id, {
        "transport": transport,
        "transport_error": transport_error,
        "read_back": confirmed,
        "read_ids": read_ids[:8],
        "check_in_id": check_in_id,
    }


def move_rows(
    repo_root: str, rows: Iterable[tuple[str, str]], base_url: str, apply: bool
) -> list[dict[str, Any]]:
    """Move each named row and read it back; the read-back is the verdict.

    ``POST {base}/tasks/update_status`` then ``GET {base}/tasks/get``. This
    instance answers ``500`` for a mutation that landed (``board.py``'s
    ``Live`` note), so a transport error is never read as a refusal.
    """
    out: list[dict[str, Any]] = []
    for row_id, target in rows:
        entry: dict[str, Any] = {"task_id": row_id, "target_status": target}
        if not apply:
            entry["transport"] = "dry-run"
            out.append(entry)
            continue
        if board is None:
            raise CheckpointError(f"board.py is not importable ({BOARD_IMPORT_ERROR})")
        load_env(repo_root)
        client = board.RestClient(base_url=base_url)
        transport = client.update_status(row_id, target)
        landed, observed = client.verify_status(row_id, target)
        entry.update({"transport": transport, "observed_status": observed, "landed": landed})
        if not landed:
            print(
                f"checkpoint: FAIL {RULE_ROW_NOT_LANDED}: {row_id} targeted {target!r} and "
                f"read back {observed!r}",
                file=sys.stderr,
            )
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected ID=STATUS, got {value!r}")
    row_id, target = value.split("=", 1)
    row_id, target = row_id.strip(), target.strip()
    if not row_id or not target:
        raise argparse.ArgumentTypeError(f"expected ID=STATUS, got {value!r}")
    return row_id, target


def resolve_measure(args: argparse.Namespace) -> dict[str, Any]:
    skipped: list[tuple[str, str]] = []
    if args.measure:
        path = args.measure
    else:
        roots: list[str] = []
        if os.environ.get("APEIRON_REG1_ROOT"):
            roots.append(os.environ["APEIRON_REG1_ROOT"])
        roots.extend(args.measure_root or [])
        if not roots:
            roots.extend(DEFAULT_MEASURE_ROOTS)
        path, skipped = newest_measure(roots)
    measure = read_measure(path)
    measure["skipped_newer"] = skipped
    return measure


def checkin_status_for(measure: dict[str, Any], override: str | None) -> str:
    """The check-in status the measurement implies, or the caller's, checked.

    ``off_track`` when the run reports ``full_acceptance=false`` or a non-zero
    failing tally, ``on_track`` when it reports a clean acceptance, ``caution``
    when the stream carries no summary to read. An explicit ``--checkin-status``
    is accepted only when the endpoint's enum contains it, so a vocabulary from
    elsewhere fails here by name instead of as an HTTP 400.
    """
    if override:
        if override not in CHECK_IN_STATUSES:
            raise CheckpointError(
                f"--checkin-status {override!r} is not in {', '.join(CHECK_IN_STATUSES)}; "
                "this endpoint's enum is measured from its own HTTP 400"
            )
        return override
    summary = measure["summary"]
    if not summary:
        return "caution"
    failing = _as_int(summary, "fail")
    full = (summary.get("full_acceptance") or "").lower()
    if full == "true" and failing in (0, None):
        return "on_track"
    if failing not in (0, None) or full == "false":
        return "off_track"
    return "caution"


def print_headline(measure: dict[str, Any], stream=None) -> None:
    stream = stream or sys.stdout
    print(f"measure {measure['path']} sha256={measure['sha256']} bytes={measure['bytes']}", file=stream)
    for path, reason in measure.get("skipped_newer") or ():
        print(f"skipped-newer {path}: {reason}", file=stream)
    summary = measure["summary_line"] or "no acceptance_summary line in this stream"
    print(f"reg1 {summary}", file=stream)
    for incumbent in HEADLINE_INCUMBENTS:
        entry = measure["headline"].get(incumbent)
        if not entry:
            print(f"hashtrinity {incumbent} NOT in this stream", file=stream)
            continue
        print(
            f"hashtrinity {incumbent} decided_pairs={_fmt(entry['decided_pairs'])} "
            f"ratio_mean={_fmt(entry['ratio_mean'])} ratio_min={_fmt(entry['ratio_min'])} "
            f"ratio_max={_fmt(entry['ratio_max'])} ratio_spread={_fmt(entry['ratio_spread'])} "
            f"digest_fail={_fmt(entry['digest_fail'])} "
            f"worst_cell={entry['worst_cell'] or '-'}@{_fmt(entry['worst_cell_ratio_max'])} "
            f"pass_at_campaign={entry['pass_at_campaign'] or '-'}",
            file=stream,
        )


def run_measure(args: argparse.Namespace) -> int:
    measure = resolve_measure(args)
    print_headline(measure)
    return 0


def run_cycle(args: argparse.Namespace, apply: bool) -> int:
    repo_root = os.path.abspath(args.repo_root)
    measure = resolve_measure(args)
    staged = staged_digest(repo_root)
    worktree = worktree_digest(repo_root)
    rows = list(args.row or [])
    text = checkin_text(measure, args.project, staged, worktree, rows)
    marker = marker_path(repo_root, args.marker)
    print_headline(measure)
    print(f"checkpoint mode={'post' if apply else 'dry-run'} project={args.project} marker={os.path.relpath(marker, repo_root)}")
    print(text)
    if rows:
        print("rows " + " ".join(f"{row_id}=>{target}" for row_id, target in rows))
    if not apply:
        print(f"checkpoint: dry-run; no write to {args.base_url}/{CHECK_IN_ENDPOINT}. Pass --post to apply.")
        payload = build_marker("dry-run", args.project, measure, staged, worktree, text, [], "")
        write_marker(marker, payload)
        print(
            f"checkpoint: wrote {os.path.relpath(marker, repo_root)} "
            f"check_in_digest={payload['check_in_digest'][:16]} staged_digest={payload['staged_digest'][:16]}"
        )
        return 0
    check_in_id, transport = post_check_in(
        repo_root,
        args.project_id,
        args.base_url,
        checkin_status_for(measure, args.checkin_status),
        text,
    )
    if not transport.get("read_back"):
        return 1
    row_records = move_rows(repo_root, rows, args.base_url, apply=True)
    unlanded = [entry for entry in row_records if not entry.get("landed", True)]
    payload = build_marker(
        "post", args.project, measure, staged, worktree, text, row_records, check_in_id
    )
    payload["check_in_transport"] = transport
    write_marker(marker, payload)
    print(
        f"checkpoint: posted {args.base_url}/{CHECK_IN_ENDPOINT} read-back="
        f"{transport.get('read_back')} check_in_id={check_in_id or 'not-returned-by-the-endpoint'} "
        f"rows={len(row_records)} unlanded={len(unlanded)}"
    )
    print(
        f"checkpoint: wrote {os.path.relpath(marker, repo_root)} "
        f"staged_digest={payload['staged_digest'][:16]} source_digest={payload['source_digest'][:16]}"
    )
    if unlanded:
        return 1
    return 0


# ---------------------------------------------------------------------------
# The fixture stream and the control suite
# ---------------------------------------------------------------------------


FIXTURE_STREAM = "\n".join(
    [
        "measurement_precondition=ok measurement_valid=true profile=full accept=true "
        "measurement_verdict=measured",
        "apeiron_head=e6e9d135b6fdf0c903b725633f7a85a0a109545a dirty=yes",
        "hashtrinity_rev=4cc14471ae24f3f204fb0b0e91736ef75c909358",
        'control=placement-only-address observed=red reason="a placement-only TableAddress is '
        'not the production address for a coordinate word"',
        "acceptance_summary registered=2752 load_bearing=2408 applicable=1500 pass=802 fail=698 "
        "missing=0 unauthorized=0 not_applicable=908 informational=344 concurrency_unmatched=6 "
        "address_face_unmatched=6 worst_pair=occ16-clustered-skewed-absent-miss-replacement/hashbrown_fxhash "
        "worst_ratio=8.2117 full_acceptance=false",
        "acceptance_repeats profile=full axis=campaign campaigns=3 decided_pairs=1482 "
        "pooled_pass=802 pass_mean=826.00 pass_min=818 pass_max=839 pass_spread=21 pass_sd=9.27",
        "acceptance_ratio incumbent=hashtrinity_table class=load-bearing pairs=160 digest_fail=0 "
        "r_min=0.0959 r_max=5.0974",
        "acceptance_ratio incumbent=hashtrinity_count_par class=load-bearing pairs=9 digest_fail=3 "
        "r_min=1.6580 r_max=5.6300",
        "acceptance_ratio incumbent=hashtrinity_get_each class=load-bearing pairs=9 digest_fail=3 "
        "r_min=1.6836 r_max=5.6612",
        "acceptance_repeats_incumbent incumbent=hashtrinity_table decided_pairs=160 ratio_mean=1.7556 "
        "ratio_min=0.0959 ratio_max=5.0974 ratio_spread=53.1660 pass_at_campaign=[64, 62, 62]",
        "acceptance_repeats_incumbent incumbent=hashtrinity_count_par decided_pairs=6 ratio_mean=3.4310 "
        "ratio_min=1.6580 ratio_max=5.6300 ratio_spread=3.3957 pass_at_campaign=[0, 0, 0]",
        "acceptance_repeats_incumbent incumbent=hashtrinity_get_each decided_pairs=6 ratio_mean=3.4299 "
        "ratio_min=1.6836 ratio_max=5.6612 ratio_spread=3.3625 pass_at_campaign=[0, 0, 0]",
        "acceptance_cell_ratio cell=occ1-direct-index-resident-present-hit-replacement "
        "incumbent=hashtrinity_table campaigns=3 ratio_mean=1.7500 ratio_min=1.7489 ratio_max=1.7517 "
        "ratio_spread=1.0016 ratio_spread_abs=0.0028 pass_at_campaign=FFF pooled=fail",
        "acceptance_cell_ratio cell=occ8-offset-beyond-direct-absent-miss-replacement "
        "incumbent=hashtrinity_table campaigns=3 ratio_mean=4.9000 ratio_min=4.8800 ratio_max=5.0974 "
        "ratio_spread=1.0446 ratio_spread_abs=0.2174 pass_at_campaign=FFF pooled=fail",
        "acceptance_cell_ratio cell=occ1-direct-index-resident-present-hit-replacement "
        "incumbent=hashtrinity_count_par campaigns=3 ratio_mean=3.4310 ratio_min=3.2000 "
        "ratio_max=5.6300 ratio_spread=1.7594 ratio_spread_abs=2.4300 pass_at_campaign=FFF pooled=fail",
        "acceptance_address_face cell=spatial/n1000 incumbent=hashtrinity_count_par face=matched "
        'reason="address fold is exact on the packed face"',
        "acceptance_address_face cell=spatial/n5000 incumbent=hashtrinity_count_par face=unmatched "
        'reason="no key store: FreqLine holds (u64, V) with no key column"',
    ]
) + "\n"


def write_fixture(path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(FIXTURE_STREAM)
    return path


def scratch_root() -> str:
    """Where a throwaway control copy lives.

    ``AGENTS.md`` §10 (the tmpfs row) keeps disposable copies off ``/tmp``:
    ``/home/socol/oc-tmp`` unless ``APEIRON_CHECKPOINT_SCRATCH`` names another.
    """
    return os.environ.get("APEIRON_CHECKPOINT_SCRATCH") or "/home/socol/oc-tmp"


HOOK = ".githooks/pre-commit"
# Every tool path the hook names, in either form it uses: a ``run_step`` line's
# ``python3 -B PATH``, or the variable the digest step assigns. Scoped to
# ``tools/`` and ``proof/`` so a ``python3`` elsewhere in the hook's prose (a fix
# line, a comment) cannot inject a path.
HOOK_TOOL = re.compile(
    r"(?:python3[ \t]+-B[ \t]+|^digest_gate=)((?:tools|proof)/[\w./-]+\.py)", re.M
)


def hook_tools(live: str) -> list[str]:
    """The tools ``.githooks/pre-commit`` runs, read out of the hook.

    The hook is the authority on what a commit runs, so the scaffold's tool set
    is read from it rather than restated here. A restated list rots *silently*:
    the scaffold still builds and the control still fails, and the diagnostic
    names a step the control never exercised. That is exactly the failure this
    replaces -- the copy list had lost ``tools/rooting/digest_recheck.py`` and
    ``tools/board/board_rows.py``, which the hook and the driver both need.
    """
    with open(os.path.join(live, HOOK), encoding="utf-8") as handle:
        return sorted(set(HOOK_TOOL.findall(handle.read())))


def tool_modules(live: str, tool: str) -> list[str]:
    """``tool`` and every sibling module beside it, as repo-relative paths.

    A Python tool in this repository is a directory's worth of modules imported
    by name (``lane.py`` over ``lane_core``/``lane_fetch``/``lane_modes``/
    ``lane_controls``/``lane_bundle``/``lane_inventory``; ``board.py`` over
    ``board_core``/``board_rows``/...), so copying one file is a partial copy:
    it fails at *import* time and surfaces as a ``None`` module with an
    ``AttributeError`` far from the omission, not as a missing path. Copying the
    directory means a new sibling cannot rot the list again.
    """
    prefix = os.path.dirname(tool)
    pattern = os.path.join(live, prefix, "*.py") if prefix else os.path.join(live, "*.py")
    return sorted(os.path.relpath(path, live) for path in glob.glob(pattern))


def governing_documents(live: str) -> dict[str, tuple[str, ...]]:
    """The stub documents ``tools/check-claims.py`` requires, from the checker.

    The checker owns the set and the headings it covers; the stubs are written
    from its own declarations, so a document or a covered heading added there is
    present in the scaffold without a second name maintained here. Each stub
    carries its covered heading and no prose, so the checker finds the section it
    requires and extracts no claim from it.
    """
    spec = importlib.util.spec_from_file_location(
        "apeiron_check_claims", os.path.join(live, "tools", "check-claims.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        name: tuple(module.COVERED_SECTIONS.get(name, ())) for name in module.GOVERNING_DOCUMENTS
    }


def build_control_repo(root: str) -> str:
    """A throwaway git work tree carrying the real hook and the real driver.

    Minimal but faithful: the same ``.githooks/pre-commit``, the same tool set
    the hook names (each with its whole module directory), the same
    ``tools/board/*.py`` this file lives in, and the smallest crate, document set
    and board snapshot each *other* step needs. The live tree is never touched:
    every path is written under ``root``.

    The hook and its tools stay in the scaffold -- they are what a commit runs,
    and the set is derived from the hook so it cannot rot -- but the controls
    here do not run it: the checkpoint step it used to carry was retired in
    ``c9a2d0c``, so ``checkpoint.py check`` is exercised as the command it is
    (see ``check``). The board snapshot is carried for the same reason: the hook's
    ``lane.py staged`` rule is its only consumer.
    """
    repo = os.path.join(root, "control-repo")
    if os.path.isdir(repo):
        subprocess.run(["rm", "-rf", repo], check=True)
    for sub in ("tools/board", "tools/hooks", ".githooks", "src", "docs", "proof", ".apeiron"):
        os.makedirs(os.path.join(repo, sub), exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    live = REPO_ROOT
    # Derived from the live tree, never restated: the hook names the tools a
    # commit runs, each tool brings its module directory, and `tools/board` is
    # both this file's own directory and the driver's, which
    # `tools/hooks/lane_core.py` imports as the `board:` rule's authority.
    relatives = {".githooks/pre-commit", ".githooks/commit-msg"}
    for tool in hook_tools(live):
        relatives.update(tool_modules(live, tool))
    relatives.update(tool_modules(live, os.path.relpath(os.path.join(here, "board.py"), live)))
    for relative in sorted(relatives):
        source = os.path.join(live, relative)
        target = os.path.join(repo, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(source, "rb") as src, open(target, "wb") as dst:
            dst.write(src.read())
        if relative.startswith(".githooks"):
            os.chmod(target, 0o755)
    # The other pre-commit steps, satisfied minimally and honestly:
    # a formatted one-function crate, the governing documents
    # `tools/check-claims.py` requires (read from the checker, empty of prose so
    # it finds no claim), and one Lean declaration so its "no declarations"
    # structural failure does not fire.
    with open(os.path.join(repo, "Cargo.toml"), "w", encoding="utf-8") as handle:
        handle.write('[package]\nname = "control"\nversion = "0.1.0"\nedition = "2021"\n')
    with open(os.path.join(repo, "src", "lib.rs"), "w", encoding="utf-8") as handle:
        handle.write("pub fn probe() -> i32 {\n    42\n}\n")
    for name, headings in governing_documents(live).items():
        target = os.path.join(repo, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("".join(f"{heading}\n" for heading in headings))
    with open(os.path.join(repo, "proof", "Control.lean"), "w", encoding="utf-8") as handle:
        handle.write("theorem control_kernel_cone : True := trivial\n")
    measure = write_fixture(os.path.join(repo, "measure.json"))
    # The snapshot is the driver's own *raw* fixture, not its normalized
    # document: `lane.py` re-normalizes a snapshot through `board.load_document`,
    # and a normalized task carries no `description`, so its plan ids and its
    # `board:` field would be gone and no declared row could resolve.
    #
    # Nothing here consumes it: no control runs the hook (see the docstring
    # above). Its one rule, `board-snapshot-stale`, is `tools/hooks/lane_core.py`'s
    # and it is unsatisfiable from a capture written before the commit it would
    # resolve -- measured on this scaffold while the controls still committed:
    # `git commit` writes the index at its own start, so the index advanced past
    # the stamp by 0.5 ms on an immediate commit and by 1.2 s on one started
    # 1.2 s after `git add`, while a no-op commit left it untouched. A control
    # asserting it would have been a vacuous pass either way, so it is recorded
    # here instead. Clearing it for a real lane means comparing whole seconds or
    # capturing from inside the commit; both are the live hook's decision.
    snapshot = os.path.join(repo, "board-snapshot.json")
    with open(snapshot, "w", encoding="utf-8") as handle:
        json.dump(board.FIXTURE_BASELINE, handle)
    env = dict(os.environ)
    env["APEIRON_HOOKS_BOARD_SOURCE"] = snapshot

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=repo, env=env, capture_output=True, text=True, timeout=300
        )

    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True, timeout=60)
    git("config", "user.email", "control@example.invalid")
    git("config", "user.name", "checkpoint control")
    git("add", "-A")
    # The first commit installs no hook: the hooks path is set after it, so the
    # base commit is not itself the control case.
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "control base"],
        cwd=repo,
        check=True,
        timeout=120,
    )
    git("config", "core.hooksPath", ".githooks")
    return repo


def check(repo: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    """``checkpoint.py check`` over the scaffold, under the harness that runs it.

    The query has no hook step: the per-cycle checkpoint step of
    ``.githooks/pre-commit`` was retired with the acceptance harness in
    ``c9a2d0c`` (owner direction 2026-10-05), so no commit output carries
    ``checkpoint: OK`` or ``checkpoint: FAIL`` and asserting on one asserts on a
    step that no longer runs. The harness that still runs is the command itself,
    so a control invokes the scaffold's copy -- this file, byte-identical -- in
    its own process with the scaffold as ``--repo-root``, which is exactly what a
    checkpoint lane runs against its own staged tree.

    The ``git add`` that precedes each call is kept: ``check`` reads the index,
    not the working tree, so a staged ``src/`` path is the input the rule is
    about. Both streams are captured because ``run_check`` prints its pass to
    stdout and its refusals to stderr. Temporal cost: one process per call,
    dominated by ``git diff --cached`` over the source prefixes.
    """
    return subprocess.run(
        [
            sys.executable,
            "-B",
            os.path.join(repo, "tools", "board", "checkpoint.py"),
            "check",
            "--repo-root",
            repo,
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def run_controls(stream=None) -> int:
    """Every control green, each seeded defect red by name, restored green.

    Temporal cost: four controls each spawn one ``checkpoint.py check`` over the
    scaffold and one runs ``dry-run``; each reads the git index once, so the suite
    is O(5E) in the staged entry count E (``git diff --cached`` plus a tree
    digest, both linear in E). Measured wall clock: 0.55-0.62 s over five runs,
    and 20.5 s once on a box already running other lanes, so the figure is a floor
    and not a ceiling.
    """
    stream = stream or sys.stdout
    failures = 0
    root = os.path.join(scratch_root(), f"checkpoint-self-test-{os.getpid()}")
    os.makedirs(root, exist_ok=True)

    # -- control 1: a quoted value survives, which a whitespace split breaks ---
    fields = parse_kv(
        'control=placement-only-address observed=red reason="a placement-only TableAddress is '
        'not the production address for a coordinate word"'
    )
    if "placement-only TableAddress" in fields.get("reason", ""):
        print("CONTROL pass kv-quotes-survive: the quoted value holds its spaces", file=stream)
    else:
        failures += 1
        print(f"CONTROL FAIL kv-quotes-survive: reason={fields.get('reason')!r}", file=stream)

    # -- control 2: the headline is the stream's own numbers ------------------
    fixture = write_fixture(os.path.join(root, "measure-fixture.json"))
    try:
        measure = read_measure(fixture)
        entry = measure["headline"].get("hashtrinity_table", {})
        checks = (
            entry.get("decided_pairs") == 160,
            abs(float(entry.get("ratio_mean", 0)) - 1.7556) < 1e-9,
            entry.get("digest_fail") == 0,
            entry.get("worst_cell") == "occ8-offset-beyond-direct-absent-miss-replacement",
            abs(float(entry.get("worst_cell_ratio_max", 0)) - 5.0974) < 1e-9,
            measure["headline"]["hashtrinity_count_par"]["digest_fail"] == 3,
            measure["address_faces"]["hashtrinity_count_par"]["unmatched"] == 1,
            "full_acceptance=false" in measure["summary_line"],
        )
        if all(checks):
            print(
                "CONTROL pass headline-read: means, digest fails, worst cell and the summary "
                "tally are the stream's own values",
                file=stream,
            )
        else:
            failures += 1
            print(f"CONTROL FAIL headline-read: {checks} for {entry}", file=stream)
        # Seeded defect: a stream with no headline line at all must be refused by
        # name rather than reduced to zeros.
        empty = os.path.join(root, "measure-empty.json")
        with open(empty, "w", encoding="utf-8") as handle:
            handle.write("measurement_precondition=ok\n")
        try:
            read_measure(empty)
            failures += 1
            print("CONTROL FAIL unreadable-stream: a stream with no headline was accepted", file=stream)
        except CheckpointError as exc:
            if RULE_MEASURE_UNREADABLE in str(exc):
                print(f"CONTROL pass unreadable-stream: {RULE_MEASURE_UNREADABLE}", file=stream)
            else:
                failures += 1
                print(f"CONTROL FAIL unreadable-stream: {exc}", file=stream)
    except CheckpointError as exc:
        failures += 1
        print(f"CONTROL FAIL headline-read: {exc}", file=stream)

    # -- control 3: the newest readable stream, with the rejects named ---------
    streams = os.path.join(root, "streams")
    os.makedirs(streams, exist_ok=True)
    write_fixture(os.path.join(streams, "run1.json"))
    # A run still being written is the newest file on disk and has no summary
    # line yet; taking it blind would report a truncated campaign as the number.
    truncated = os.path.join(streams, "run2.json")
    with open(truncated, "w", encoding="utf-8") as handle:
        handle.write("measurement_precondition=ok profile=full\n")
    now = time.time()
    os.utime(os.path.join(streams, "run1.json"), (now - 600, now - 600))
    os.utime(truncated, (now, now))
    chosen, skipped = newest_measure([streams])
    if chosen.endswith("run1.json") and len(skipped) == 1 and skipped[0][0] == truncated:
        print(
            "CONTROL pass newest-readable: the truncated newer stream was skipped by name "
            f"({os.path.basename(truncated)}) and {os.path.basename(chosen)} taken",
            file=stream,
        )
    else:
        failures += 1
        print(
            f"CONTROL FAIL newest-readable: chose {chosen}, skipped {skipped}", file=stream
        )

    # -- controls 5-8: the query, under the harness that runs it ----------------
    # The harness is `checkpoint.py check` itself, not a commit: the hook step
    # that used to carry it was retired with the acceptance harness in
    # `c9a2d0c`, so a commit's output cannot carry `checkpoint: OK`/`FAIL` and
    # asserting on one asserts on a step that does not run. Each control keeps
    # its `git add`, because the rule is about the *staged* tree.
    try:
        repo = build_control_repo(root)
    except (OSError, subprocess.SubprocessError, AssertionError) as exc:
        failures += 1
        print(f"CONTROL FAIL control-repo: {type(exc).__name__}: {exc}", file=stream)
        repo = ""
    if repo:
        env = dict(os.environ)
        lib = os.path.join(repo, "src", "lib.rs")
        marker = os.path.join(repo, ".apeiron", "checkpoint.json")

        # (5) a staged src/ change with no marker at all must refuse, and the
        #     refusal must name the marker and the command that writes it.
        with open(lib, "a", encoding="utf-8") as handle:
            handle.write("\npub fn seeded() -> i32 {\n    43\n}\n")
        subprocess.run(["git", "add", "src/lib.rs"], cwd=repo, check=True, timeout=60)
        result = check(repo, env)
        output = result.stdout + result.stderr
        if (
            result.returncode != 0
            and "checkpoint: FAIL" in output
            and RULE_NO_MARKER in output
            and ".apeiron/checkpoint.json" in output
            and "checkpoint.py dry-run" in output
        ):
            print(
                f"CONTROL pass no-marker-refuses: exit {result.returncode}, {RULE_NO_MARKER} "
                "named with the marker path and the fix",
                file=stream,
            )
        else:
            failures += 1
            print(
                f"CONTROL FAIL no-marker-refuses: exit {result.returncode}\n{output}",
                file=stream,
            )

        # (6) a marker written for a *different* staged tree must refuse, naming
        #     both digests. Written from this module's own repository, so it
        #     records a digest the scaffold's index cannot match.
        write_marker(
            marker,
            build_marker(
                "dry-run",
                "apeiron",
                read_measure(fixture),
                staged_digest(REPO_ROOT),
                worktree_digest(REPO_ROOT),
                "seeded stale marker",
                [],
                "",
            ),
        )
        result = check(repo, env)
        output = result.stdout + result.stderr
        # The refusal must name the rule, the marker it read (by the digest of
        # the check-in text it carries) and the digest it compared against.
        stale_identity = sha256_of("seeded stale marker")[:16]
        recorded = staged_digest(REPO_ROOT)[0][:16]
        if (
            result.returncode != 0
            and RULE_STALE_MARKER in output
            and f"check-in {stale_identity}" in output
            and f"records staged_digest={recorded}" in output
        ):
            print(
                f"CONTROL pass stale-marker-refuses: exit {result.returncode}, "
                f"{RULE_STALE_MARKER} named with both digests",
                file=stream,
            )
        else:
            failures += 1
            print(f"CONTROL FAIL stale-marker-refuses: exit {result.returncode}\n{output}", file=stream)

        # (7) with a marker written for *this* staged tree the same query passes.
        # `pre-commit: OK` is dropped: it belonged to the hook's own summary,
        # and no hook step runs this query any more.
        os.remove(marker)
        dry = subprocess.run(
            [
                sys.executable,
                "-B",
                os.path.join(repo, "tools", "board", "checkpoint.py"),
                "dry-run",
                "--measure",
                os.path.join(repo, "measure.json"),
                "--repo-root",
                repo,
            ],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        result = check(repo, env)
        output = result.stdout + result.stderr
        if dry.returncode == 0 and result.returncode == 0 and "checkpoint: OK" in output:
            print(
                f"CONTROL pass fresh-marker-passes: dry-run exit {dry.returncode}, check exit "
                f"{result.returncode} with `checkpoint: OK`",
                file=stream,
            )
        else:
            failures += 1
            print(
                f"CONTROL FAIL fresh-marker-passes: dry-run exit {dry.returncode}, check exit "
                f"{result.returncode}\n{dry.stdout}\n{dry.stderr}\n{output}",
                file=stream,
            )

        # (8) restore: the marker is now stale for the *next* src edit, so the
        #     refusal returns -- the check is not a one-shot pass.
        with open(lib, "a", encoding="utf-8") as handle:
            handle.write("\npub fn restored() -> i32 {\n    44\n}\n")
        subprocess.run(["git", "add", "src/lib.rs"], cwd=repo, check=True, timeout=60)
        result = check(repo, env)
        output = result.stdout + result.stderr
        if result.returncode != 0 and RULE_STALE_MARKER in output:
            print(
                f"CONTROL pass staleness-returns: exit {result.returncode}, "
                f"{RULE_STALE_MARKER} on the next src edit",
                file=stream,
            )
        else:
            failures += 1
            print(
                f"CONTROL FAIL staleness-returns: exit {result.returncode}\n{output}", file=stream
            )

    total = 8
    # The throwaway work tree is pruned here, on every path: every diagnostic the
    # controls need is in the printed report, and a suite that left a ~570 kB
    # clone per run under `/home/socol/oc-tmp` leaked one copy per invocation
    # (15 accumulated over this lane's runs; AGENTS.md §10, the tmpfs row, and
    # the checkpoint-obligation lane's own prune step).
    shutil.rmtree(root, ignore_errors=True)
    if failures:
        print(f"CONTROL FAIL suite: {failures} of {total} controls failed", file=stream)
        return 2
    print(f"CONTROL pass suite: {total}/{total} controls", file=stream)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="checkpoint.py",
        description="Measure, post the board check-in, move the rows, and write the marker.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="check",
        choices=("measure", "dry-run", "post", "check", "self-test"),
        help="measure | dry-run | post | check | self-test",
    )
    parser.add_argument("--measure", metavar="PATH", help="reg1 acceptance stream (default: newest under the stated roots)")
    parser.add_argument(
        "--measure-root",
        action="append",
        metavar="DIR",
        help=f"root searched for the newest *.json (default: $APEIRON_REG1_ROOT else {' '.join(DEFAULT_MEASURE_ROOTS)})",
    )
    parser.add_argument("--repo-root", default=REPO_ROOT, help="repository the digests and the marker are read against")
    parser.add_argument("--marker", metavar="PATH", help=f"marker path (default: {DEFAULT_MARKER})")
    parser.add_argument("--project", default="apeiron", help="project name the check-in names")
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID, help="operately project id for --post")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="operately external v1 API base")
    parser.add_argument(
        "--checkin-status",
        help=f"check-in status override (one of {', '.join(CHECK_IN_STATUSES)}; default derived from the measurement)",
    )
    parser.add_argument("--row", action="append", type=parse_assignment, metavar="ID=STATUS", help="row to move (repeatable)")
    parser.add_argument("--self-test", action="store_true", help="seeded-defect control suite")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test or args.mode == "self-test":
        return run_controls()
    try:
        if args.mode == "check":
            return run_check(args)
        if args.mode == "measure":
            return run_measure(args)
        return run_cycle(args, apply=args.mode == "post")
    except CheckpointError as exc:
        print(f"checkpoint.py: {exc}", file=sys.stderr)
        return 2
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"checkpoint.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())