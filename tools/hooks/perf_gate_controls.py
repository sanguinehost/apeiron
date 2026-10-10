#!/usr/bin/env python3
"""perf_gate_controls: the pre-commit perf cost-model gate and its self-test.

Owner requirement (2026-10-08): it must not be possible to land a performance
improvement without formalising, in the cost model, how and why it works. A
staged change under a performance source path must carry, in the same commit:

  * a new P-row in ``docs/evidence/perf-predictions.md``, and
  * that row's ``Lean`` cell naming a dotted declaration that resolves under
    ``proof/corpus/Perf/`` and is added or changed in the staged diff under that
    directory.

The parser, the row/column rules and the declaration resolution are taken from
``tools/check-claims.py`` so the hook and the claim checker cannot drift. There
is no bypass flag and no environment override: the step refuses, naming the
staged perf path and the missing piece. A staged diff of a perf path that adds and
removes only ``//`` comment or blank lines changes no cost and is not a perf change;
any other changed line keeps the obligation.

Modes::

    python3 -B tools/hooks/perf_gate_controls.py --staged [--root DIR]
        run the gate against the staged index of the repository at ``--root``
        (default: the current directory). Exit 0 accept, 1 refuse.

    python3 -B tools/hooks/perf_gate_controls.py --self-test
        stage fixtures in throwaway git repositories under a temp directory, run
        the gate, and report each exit status.

``.githooks/pre-commit`` calls the ``--staged`` mode; print ``SKIP`` with the
reason when no perf path is staged.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Declared performance source paths. A staged path under any of these is a perf
# change and carries the formalisation obligation.
PERF_PREFIXES = ("src/simt_nn/", "src/gpu/", "src/spirv/", "src/gpu/decode/")


def _load_check_claims():
    path = Path(__file__).resolve().parents[1] / "check-claims.py"
    spec = importlib.util.spec_from_file_location("apeiron_check_claims", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CC = _load_check_claims()


# --------------------------------------------------------------------------
# git plumbing (staged index only)
# --------------------------------------------------------------------------

def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True
    )


def staged_paths(root):
    proc = git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def file_at(root, rev, path):
    proc = git(root, "show", f"{rev}:{path}")
    if proc.returncode != 0:
        return None
    return proc.stdout


def is_perf_path(path):
    normalized = path.replace("\\", "/")
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in PERF_PREFIXES
    )


def is_comment_only(root, path):
    """True when the staged diff of ``path`` adds and removes only ``//`` comment or blank lines.

    A comment cannot change a cost, so it carries no formalisation obligation. Conservative: any
    other changed line (code, a block comment delimiter, a trailing comment) makes the change
    non-comment, and a file new to the index has only added code lines.
    """
    proc = git(root, "diff", "--cached", "-U0", "--", path)
    changed = [
        line[1:].strip()
        for line in proc.stdout.splitlines()
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    return bool(changed) and all(not text or text.startswith("//") for text in changed)


def perf_rows(text):
    """``(id, raw_line)`` for every P-row (header-ful and header-less alike)."""
    rows = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.count("|") >= 2):
            continue
        first = stripped.strip("|").split("|")[0].strip()
        if CC.PERF_ROW_ID_RE.match(first):
            rows.append((first, line))
    return rows


def staged_row_lean(staged_text, row_id):
    """The Lean cell of ``row_id`` in a staged file: ``(lean, header_less)``."""
    for block in CC.iter_blocks(staged_text.splitlines()):
        if block["kind"] != "table":
            continue
        header = None
        lean_column = None
        for _, raw in block["lines"]:
            stripped = raw.strip()
            if CC.TABLE_SEP_RE.match(stripped):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if header is None:
                lowered = [cell.lower() for cell in cells]
                if cells and cells[0].lower() == "id" and CC.PERF_LEAN_COLUMN in lowered:
                    header = cells
                    lean_column = lowered.index(CC.PERF_LEAN_COLUMN)
                    continue
            if not cells or not CC.PERF_ROW_ID_RE.match(cells[0]) or cells[0] != row_id:
                continue
            if header is not None:
                if lean_column is not None and len(cells) > lean_column:
                    return cells[lean_column].replace("`", "").strip(), False
                return "", False
            return CC.perf_lean_field(raw), True
    return None, None


def staged_perf_index(root):
    """Declarations and namespaces from the staged blobs under proof/corpus/Perf/."""
    declarations = set()
    namespaces = set()
    proc = git(root, "ls-files", "--", str(CC.PERF_CORPUS))
    for path in proc.stdout.splitlines():
        if not path.endswith(".lean"):
            continue
        blob = file_at(root, "", path)
        if blob is None:
            continue
        declarations.update(CC.DECL_RE.findall(blob))
        namespaces.update(CC.NAMESPACE_RE.findall(blob))
    return declarations, namespaces


def staged_added_lines(root, prefix):
    proc = git(root, "diff", "--cached", "--", prefix)
    return [
        line[1:]
        for line in proc.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def refuse(message):
    print(f"perf cost-model: FAIL {message}", file=sys.stderr)
    return 1


def run_gate(root):
    root = Path(root).resolve()
    paths = staged_paths(root)
    perf = [
        path
        for path in paths
        if is_perf_path(path) and not is_comment_only(root, path)
    ]
    if not perf:
        print(
            "perf cost-model: SKIP (no staged code change under "
            f"{', '.join(PERF_PREFIXES)}; {len(paths)} staged path(s))"
        )
        return 0

    staged_text = file_at(root, "", CC.PERF_PREDICTIONS)
    if staged_text is None:
        return refuse(
            f"staged perf path {perf[0]} but {CC.PERF_PREDICTIONS} is not staged"
        )
    base_text = file_at(root, "HEAD", CC.PERF_PREDICTIONS) or ""
    base_ids = {row_id for row_id, _ in perf_rows(base_text)}
    staged_ids = {row_id for row_id, _ in perf_rows(staged_text)}
    added = sorted(staged_ids - base_ids)
    if not added:
        return refuse(
            f"staged perf path {perf[0]} adds no P-row to {CC.PERF_PREDICTIONS}"
        )

    declarations, namespaces = staged_perf_index(root)
    perf_added = staged_added_lines(root, str(CC.PERF_CORPUS))
    for row_id in added:
        lean, header_less = staged_row_lean(staged_text, row_id)
        if not lean:
            return refuse(
                f"new P-row {row_id} has no Lean cell (a header-less row carries the "
                f"field `Lean: <decl>`)"
            )
        if lean.startswith("open:"):
            return refuse(
                f"new P-row {row_id} defers with `open:`; a theorem resolving under "
                f"{CC.PERF_CORPUS}/ is required"
            )
        if "." not in lean or not CC.lean_resolves(lean, declarations, namespaces, set()):
            return refuse(
                f"new P-row {row_id} Lean decl `{lean}` does not resolve under "
                f"{CC.PERF_CORPUS}/"
            )
        short = lean.split(".")[-1]
        if not any(short in line for line in perf_added):
            return refuse(
                f"new P-row {row_id} Lean decl `{lean}` is not added or changed in the "
                f"staged diff under {CC.PERF_CORPUS}/"
            )
    print(
        f"perf cost-model: PASS (perf path {perf[0]}; new P-row(s) {', '.join(added)}; "
        f"Lean decl resolves and is added under {CC.PERF_CORPUS}/)"
    )
    return 0


# --------------------------------------------------------------------------
# self-test (throwaway git repositories)
# --------------------------------------------------------------------------

BASE_PERF_TABLE = (
    "# predictions\n\n"
    "| id | case | cost equation | prediction | refuted if | measured | verdict | Lean |\n"
    "|---|---|---|---|---|---|---|---|\n"
    "| P1 | base lever | eq | p | if | measured | held | `Perf.Moves.base_gain` |\n"
)
BASE_PERF_LEAN = "namespace Perf.Moves\n\ntheorem base_gain : True := trivial\n"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_repo(path):
    _write(path / "src" / "simt_nn" / "norm.rs", "// norm\n")
    _write(path / "docs" / "evidence" / "perf-predictions.md", BASE_PERF_TABLE)
    _write(path / "docs" / "evidence" / "perf-open-allowlist.txt", "# base\n")
    _write(path / "proof" / "corpus" / "Perf" / "Moves.lean", BASE_PERF_LEAN)
    git(path, "init", "-q")
    git(path, "config", "user.email", "perf-gate@example.invalid")
    git(path, "config", "user.name", "perf gate control")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    return path


def _stage(repo):
    git(repo, "add", "-A")
    return repo


def _touch_perf(repo, text="pub fn perf_change() {}\n"):
    with (repo / "src" / "simt_nn" / "norm.rs").open("a", encoding="utf-8") as handle:
        handle.write(text)


def _add_perf_lean(repo, declaration="theorem brand_new_gain : True := trivial\n"):
    with (repo / "proof" / "corpus" / "Perf" / "Moves.lean").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("\n" + declaration)


def _append_prediction(repo, text):
    with (repo / "docs" / "evidence" / "perf-predictions.md").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(text)


def case_positive(repo):
    _touch_perf(repo)
    _add_perf_lean(repo)
    _append_prediction(
        repo,
        "| P2 | new lever | eq | p | if | measured | held | `Perf.Moves.brand_new_gain` |\n",
    )
    return _stage(repo)


def case_comment_only(repo):
    _touch_perf(repo, "// a comment, no code\n")
    return _stage(repo)


def case_neg_e_comment_hides_code(repo):
    _touch_perf(repo, "// a comment\npub fn perf_change() {} // trailing\n")
    return _stage(repo)


def case_neg_a_perf_only(repo):
    _touch_perf(repo)
    return _stage(repo)


def case_neg_b_new_open(repo):
    _touch_perf(repo)
    _append_prediction(
        repo, "| P2 | new lever | eq | p | if | measured | held | open: deferred |\n"
    )
    return _stage(repo)


def case_neg_c_headerless_no_lean(repo):
    _touch_perf(repo)
    _append_prediction(repo, "\n| P3 | new lever | eq | p |\n")
    return _stage(repo)


def case_neg_d_unresolved_decl(repo):
    _touch_perf(repo)
    _add_perf_lean(repo, "theorem another_gain : True := trivial\n")
    _append_prediction(
        repo,
        "| P2 | new lever | eq | p | if | measured | held | `Perf.Moves.does_not_exist` |\n",
    )
    return _stage(repo)


CASES = (
    (
        "positive: perf path + P-row + resolving decl + Perf diff -> accept",
        case_positive,
        0,
        ("PASS",),
    ),
    (
        "negative a: perf path only -> refuse",
        case_neg_a_perf_only,
        1,
        ("adds no P-row",),
    ),
    (
        "negative b: new open: P-row -> refuse",
        case_neg_b_new_open,
        1,
        ("defers with `open:`",),
    ),
    (
        "negative c: header-less row with no Lean field -> refuse",
        case_neg_c_headerless_no_lean,
        1,
        ("has no Lean cell",),
    ),
    (
        "negative d: unresolved decl -> refuse",
        case_neg_d_unresolved_decl,
        1,
        ("does not resolve under",),
    ),
    (
        "comment-only change in a perf path -> skip",
        case_comment_only,
        0,
        ("SKIP",),
    ),
    (
        "negative e: comment lines beside a code line -> refuse",
        case_neg_e_comment_hides_code,
        1,
        ("adds no P-row",),
    ),
    (
        "restore: positive re-applied -> accept",
        case_positive,
        0,
        ("PASS",),
    ),
)


def self_test():
    workdir = Path(tempfile.mkdtemp(prefix="perf-gate-selftest-"))
    results = []
    try:
        base = build_repo(workdir / "base")
        for name, mutate, expected, needles in CASES:
            slug = re.sub(r"[^A-Za-z0-9]+", "-", name)[:40]
            repo = workdir / ("case-" + slug)
            shutil.copytree(base, repo)
            mutate(repo)
            proc = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()),
                 "--staged", "--root", str(repo)],
                capture_output=True,
                text=True,
            )
            output = proc.stdout + proc.stderr
            passed = proc.returncode == expected and all(
                needle in output for needle in needles
            )
            results.append((name, expected, proc.returncode, passed, output.strip()))
        for name, expected, observed, passed, output in results:
            verdict = "PASS" if passed else "FAIL"
            print(f"[perf-gate] {name}: expected exit {expected}, observed {observed} -> {verdict}")
            for line in output.splitlines()[:2]:
                print(f"            {line}")
        failed = [name for name, _, _, passed, _ in results if not passed]
        print(
            f"\n[perf-gate] {len(results) - len(failed)}/{len(results)} "
            "controls behaved as required"
        )
        if failed:
            print("[perf-gate] FAILED: " + ", ".join(failed))
            return 1
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root (default: cwd)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--staged", action="store_true", help="run the gate against the staged index"
    )
    group.add_argument(
        "--self-test", action="store_true", help="run throwaway-repo controls"
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    return run_gate(args.root)


if __name__ == "__main__":
    sys.exit(main())
