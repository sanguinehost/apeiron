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
staged perf path and the missing piece.

A staged perf path that carries no code change is not a perf change. The staged
content is compared with its base blob (the rename source when git reports the
path as a rename destination) after normalising, token for token, the rewrites a
file move forces and nothing else:

  * the module/path token substitutions of the staged rename map (``crate::simt``
    -> ``crate::gpu::simt``, ``src/spirv`` -> ``src/gpu/spirv``, from the actual
    ``src/{ => gpu}/simt`` and ``src/{ => gpu}/spirv`` renames);
  * visibility qualifiers (``pub``, ``pub(crate)``, ``pub(super)``,
    ``pub(in ...)``), which cannot change an emitted kernel's cost; this subsumes
    the ``pub(super)`` -> ``pub(crate)`` widening a file move forces;
  * the leading ``../`` depth of an ``include_bytes!``/``include_str!`` path,
    which is relative to the moved file;
  * whitespace, line breaks and commas; and
  * non-code lines: blank, ``//`` comment, import, module declaration, attribute.

Any surviving token difference keeps the obligation, including a changed token in
the same hunk as a path rewrite. A pure rename (identical bytes) is a no-op, and a
path new to the index is a no-op only when it holds no executable token.

Modes::

    python3 -B tools/hooks/perf_gate_controls.py --staged [--root DIR]
        run the gate against the staged index of the repository at ``--root``
        (default: the current directory). Exit 0 accept, 1 refuse.

    python3 -B tools/hooks/perf_gate_controls.py --self-test
        stage fixtures in throwaway git repositories under a temp directory, run
        the gate, and report each exit status.

``.githooks/pre-commit`` calls the ``--staged`` mode; print ``SKIP`` with the
reason when no perf path is staged.

Cost: O(S + B) per run, S the staged diff bytes git emits for the rename map and
path list, and B the bytes of the staged perf blobs compared against their base.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Declared performance source paths. A staged path under any of these is a perf
# change and carries the formalisation obligation.
PERF_PREFIXES = ("src/gpu/",)


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


def blob_at(root, rev, path):
    """Raw bytes of ``rev:path`` (``rev`` empty means the index), or None when absent."""
    proc = subprocess.run(
        ["git", "-C", str(root), "show", f"{rev}:{path}"], capture_output=True
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def file_at(root, rev, path):
    data = blob_at(root, rev, path)
    return None if data is None else data.decode("utf-8", "replace")


def is_perf_path(path):
    normalized = path.replace("\\", "/")
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in PERF_PREFIXES
    )


def staged_renames(root):
    """``new_path -> old_path`` for every rename git detects in the staged index."""
    proc = subprocess.run(
        ["git", "-C", str(root), "diff", "--cached", "-M", "--name-status", "-z"],
        capture_output=True,
    )
    fields = proc.stdout.decode("utf-8", "replace").split("\0")
    renames = {}
    i = 0
    while i < len(fields):
        status = fields[i]
        if not status:
            i += 1
        elif status[0] in "RC":
            if status[0] == "R" and i + 2 < len(fields):
                renames[fields[i + 2]] = fields[i + 1]
            i += 3
        else:
            i += 2
    return renames


def module_path(path):
    """``src/gpu/simt/mod.rs`` -> ``crate::gpu::simt``; None for a non-``src`` Rust path."""
    if not (path.startswith("src/") and path.endswith(".rs")):
        return None
    rel = path[4:-3]
    if rel == "mod":
        rel = ""
    elif rel.endswith("/mod"):
        rel = rel[:-4]
    return "crate::" + rel.replace("/", "::")


def substitutions(renames):
    """Ordered ``old -> new`` token pairs derived from the staged rename map.

    Covers the exact path, the ``crate::`` module path of a ``.rs`` rename, and a directory
    prefix whose every rename agrees on the destination directory. Longest token first so an
    exact path wins over its directory.
    """
    pairs = []
    for new, old in renames.items():
        pairs.append((old, new))
        old_mod, new_mod = module_path(old), module_path(new)
        if old_mod is not None and new_mod is not None:
            pairs.append((old_mod, new_mod))
    moved = {}
    for new, old in renames.items():
        moved.setdefault(os.path.dirname(old), set()).add(os.path.dirname(new))
    for old_dir, new_dirs in moved.items():
        if len(new_dirs) == 1:
            new_dir = next(iter(new_dirs))
            if old_dir and new_dir and old_dir != new_dir:
                pairs.append((old_dir + "/", new_dir + "/"))
    pairs.sort(key=lambda pair: -len(pair[0]))
    table = {}
    for old, new in pairs:
        table.setdefault(old, new)
    return table


def normaliser(table):
    """A one-pass substituter: a token is replaced at most once, so pairs cannot chain."""
    pattern = None
    if table:
        pattern = re.compile("|".join(re.escape(token) for token in table))

    def normalise(line):
        line = _expand_braces(line)
        if pattern is None:
            return line
        return pattern.sub(lambda m: table[m.group(0)], line)

    return normalise


BRACE = re.compile(
    r"([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*)/\{([A-Za-z0-9_,.\-]+)\}([A-Za-z0-9_.\-]*)"
)


def _expand_braces(line):
    """``src/table/{a,b}.rs`` -> ``src/table/a.rs src/table/b.rs`` so each path normalises."""

    def repl(match):
        base, names, suffix = match.groups()
        return " ".join(f"{base}/{name}{suffix}" for name in names.split(","))

    return BRACE.sub(repl, line)


WIRING = re.compile(
    r"^(?:#!?\[|use\b|pub\s+use\b|pub\([^)]*\)\s+use\b|mod\b|pub\s+mod\b|"
    r"pub\([^)]*\)\s+mod\b|extern\s+crate\b)"
)
VISIBILITY = re.compile(r"\bpub\s*(?:\([^)]*\))?")
INCLUDE = re.compile(r'((?:include_bytes!|include_str!)\s*\(\s*")(?:\.\./)+')
TOKEN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def code_tokens(data, normalise):
    """Executable token stream of ``data``; None for a binary blob.

    Lines that cannot change an emitted kernel's cost are dropped: blank, ``//`` comment,
    import, module declaration and attribute. The rewrites a file move forces are normalised:
    path/module tokens, ``pub(super)`` visibility and ``include_*!`` relative depth. Whitespace,
    line breaks and commas are not tokens, so a reformat is not a change.
    """
    if b"\0" in data:
        return None
    tokens = []
    for line in data.decode("utf-8", "replace").splitlines():
        text = line.strip()
        if not text or text.startswith("//") or WIRING.match(text):
            continue
        text = VISIBILITY.sub("", text)
        text = INCLUDE.sub(r"\1", text)
        text = normalise(text)
        tokens.extend(t for t in TOKEN.findall(text) if t != ",")
    return tokens


def is_no_cost_change(root, path, renames, normalise):
    """True when ``path``'s staged content carries no code change against its base blob.

    The base blob is the rename source when the staged path is a rename destination. A path new
    to the index carries no change only when it holds no executable token.
    """
    staged = blob_at(root, "", path)
    if staged is None:
        return False
    base = blob_at(root, "HEAD", renames.get(path, path))
    if base is None:
        return code_tokens(staged, normalise) == []
    if base == staged:
        return True
    base_tokens = code_tokens(base, normalise)
    return base_tokens is not None and base_tokens == code_tokens(staged, normalise)


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
    renames = staged_renames(root)
    normalise = normaliser(substitutions(renames))
    perf = [
        path
        for path in paths
        if is_perf_path(path) and not is_no_cost_change(root, path, renames, normalise)
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
    _write(
        path / "src" / "gpu" / "nn" / "norm.rs",
        "use crate::simt::Carrier;\n\nconst N: u32 = 4;\n\nfn helper() -> u32 { N }\n",
    )
    _write(path / "src" / "simt" / "mod.rs", "pub struct Carrier;\n")
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
    with (repo / "src" / "gpu" / "nn" / "norm.rs").open("a", encoding="utf-8") as handle:
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


def _move_simt(repo, body):
    """Rename ``src/simt`` -> ``src/gpu/simt`` and rewrite ``norm.rs`` to ``body``."""
    (repo / "src" / "gpu" / "simt").mkdir(parents=True, exist_ok=True)
    git(repo, "mv", "src/simt/mod.rs", "src/gpu/simt/mod.rs")
    (repo / "src" / "gpu" / "nn" / "norm.rs").write_text(body, encoding="utf-8")
    return _stage(repo)


def case_path_rewrite_only(repo):
    return _move_simt(
        repo,
        "use crate::gpu::simt::Carrier;\n\nconst N: u32 = 4;\n\nfn helper() -> u32 { N }\n",
    )


def case_neg_path_rewrite_plus_code(repo):
    return _move_simt(
        repo,
        "use crate::gpu::simt::Carrier;\n\nconst N: u32 = 8;\n\nfn helper() -> u32 { N }\n",
    )


def case_visibility_only(repo):
    (repo / "src" / "gpu" / "nn" / "norm.rs").write_text(
        "use crate::simt::Carrier;\n\nconst N: u32 = 4;\n\npub(crate) fn helper() -> u32 { N }\n",
        encoding="utf-8",
    )
    return _stage(repo)


def case_neg_visibility_plus_code(repo):
    (repo / "src" / "gpu" / "nn" / "norm.rs").write_text(
        "use crate::simt::Carrier;\n\nconst N: u32 = 4;\n\npub(crate) fn helper() -> u32 { N + 1 }\n",
        encoding="utf-8",
    )
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
        "path-only rewrite: module rename + rewritten reference -> skip",
        case_path_rewrite_only,
        0,
        ("SKIP",),
    ),
    (
        "negative f: path rewrite plus one changed token -> refuse",
        case_neg_path_rewrite_plus_code,
        1,
        ("adds no P-row",),
    ),
    (
        "visibility-only change in a perf path -> skip",
        case_visibility_only,
        0,
        ("SKIP",),
    ),
    (
        "negative g: visibility plus one cost token -> refuse",
        case_neg_visibility_plus_code,
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
