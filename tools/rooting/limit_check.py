#!/usr/bin/env python3
# Row: `onebit-w18-literature-and-BpZokAJjBgG5iA9sXJKen4` (LEARN-ONEBIT-W18).
"""Refuse an ungraded limit claim in the one-bit record.

A limit claim is a unit of text that uses LIMIT vocabulary (converse, floor, lower bound, cannot reach,
unreachable, impossible, unprecedented, state of the art) outside inline code. It passes when:
  - the unit, or one of its enclosing headings, carries a backticked grade label in GRADES; or
  - the unit says `proved` and names, in backticks, a theorem under proof/ whose signature binds no
    hypothesis `(h… :` (so the bound is not assumed).
A registered model floor that a committed receipt has beaten (FLOORS) is no longer a floor: every unit
quoting its literal next to "floor" must carry `refuted`.

Lean (LEAN_SCOPE): a theorem docstring with LIMIT vocabulary over a signature that binds an `h…`
hypothesis must state that the bound is conditional: a grade word (anchor, model, conditional, hypothesis)
or a premise marker (under, if, when, given, assuming, "for `…`"). The markdown `proved` pass is stricter:
it admits only a theorem with no `h…` binder, so prose cannot cite a conditional bound as unconditional.

Units: a markdown list item with its continuation lines, or a paragraph; a heading is its own unit.
The label is checked per unit, not per sentence; a long unit with one label passes every sentence in it.
The checker establishes that a grade is stated, never that it is correct.

Modes:
  --check       the scope below; exit 1 on any failure
  --staged      every docs/ and book/ *.md in the index: fail on a failing unit that touches an added line
  --self-test   fixtures: each bad fixture red, each good fixture green
  --controls    mutate temporary copies of the live files: baseline green, each mutation red

A staged hunk whose removed and added lines are equal after normalising the module/path token
substitutions of the staged rename map (``src/simt`` -> ``src/gpu/simt``,
``src/table.rs`` -> ``src/map/table.rs`` and their ``crate::`` module forms) changes no claim, so
its added lines are not checked. A hunk that changes any other token is checked in full, including
the path lines beside the change. A pure rename carries no hunk and is therefore not a claim change.

Cost: O(B + T + D) per run, B the bytes of the scanned files (one pass per file to split units, one regex
pass per unit), T the bytes of proof/**/*.lean read once to index theorem signatures, and D the
staged diff bytes (the rename map and the docs hunks).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

LIMIT = re.compile(
    r"\b(converse|floors?|lower bounds?|cannot reach|can't reach|unreachable|impossible|"
    r"unprecedented|state of the art|state-of-the-art)\b",
    re.I,
)
GRADES = ("anchor", "model", "hypothesis", "conditional", "refuted")
LABEL = re.compile(r"`(" + "|".join(GRADES) + r")`")
LEAN_LABEL = re.compile(r"\b(anchor|model|conditional|hypothesis|under|if|when|given|assum\w*)\b|\bfor `", re.I)
CODE = re.compile(r"`[^`]*`")
HYP = re.compile(r"\(\s*h\w*\s*:")

MD_SCOPE = ["docs/evidence/onebit-*.md", "docs/epics/ternary-hadamard.md"]
LEAN_SCOPE = [
    "proof/corpus/Perf/CostCorrespondence.lean",
    "proof/corpus/Perf/KLTilt.lean",
    "proof/corpus/Perf/BitsRefine.lean",
]
# literal, receipt, JSON key path to the measured value that sits below it
FLOORS = [
    ("0.3887", "docs/evidence/receipts/onebit/t3-nb-3seed.json", ["variants", "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp-nb",
      "kl_insample_stats", "mean"]),
]


def theorem_index(root):
    """name -> True when its signature binds no h-hypothesis."""
    idx = {}
    decl = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+)?(?:theorem|lemma)\s+(\S+)(.*?):=", re.S | re.M)
    for d in ("proof/corpus", "proof/canonical"):
        base = root / d
        if not base.is_dir():
            continue
        for p in base.rglob("*.lean"):
            if ".lake" in p.parts:
                continue
            for m in decl.finditer(p.read_text(errors="replace")):
                idx[m.group(1).split(".")[-1]] = not HYP.search(m.group(2))
    return idx


def md_units(text):
    """Yield (first_line, last_line, unit_text, headings) with 1-based line numbers."""
    lines = text.split("\n")
    heads = {}  # level -> heading text
    cur, start, fence = [], 0, False

    def flush(end):
        if cur:
            yield start, end, "\n".join(cur), [heads[k] for k in sorted(heads)]

    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fence = not fence
            yield from flush(i - 1)
            cur = []
            continue
        if fence:
            continue
        h = re.match(r"^(#{1,6})\s", line)
        item = re.match(r"^\s*(?:[-*+]|\d+\.)\s", line)
        if h or not line.strip() or item:
            yield from flush(i - 1)
            cur = []
        if h:
            lvl = len(h.group(1))
            heads = {k: v for k, v in heads.items() if k < lvl}
            heads[lvl] = line
            yield i, i, line, [heads[k] for k in sorted(heads) if k < lvl]
            continue
        if line.strip():
            if not cur:
                start = i
            cur.append(line)
    yield from flush(len(lines))


def floor_literals(root):
    out = []
    for lit, rec, keys in FLOORS:
        v = json.loads((root / rec).read_text())
        for k in keys:
            v = v[k]
        if float(v) < float(lit):
            out.append((lit, rec, v))
    return out


def check_md(path, text, idx, beaten):
    fails = []
    for a, b, unit, heads in md_units(text):
        bare = CODE.sub("", unit)
        for lit, rec, v in beaten:
            if lit in unit and re.search(r"\bfloor", bare, re.I) and "`refuted`" not in unit:
                fails.append((a, b, f"floor {lit} is beaten by {rec} ({v:.5f}); unit needs `refuted`"))
        m = LIMIT.search(bare)
        if not m:
            continue
        if LABEL.search(unit) or any(LABEL.search(h) for h in heads):
            continue
        if "`proved`" in unit and any(
            idx.get(n.split(".")[-1]) for n in re.findall(r"`([A-Za-z_][\w.']*)`", unit)
        ):
            continue
        fails.append((a, b, f"'{m.group(0)}' without a grade label or an unconditional theorem"))
    return [(str(path), a, b, why) for a, b, why in fails]


def check_lean(path, text):
    fails = []
    pat = re.compile(r"/--(.*?)-/\s*(?:@\[[^\]]*\]\s*)?(?:theorem|lemma)\s+(\S+)(.*?):=", re.S)
    for m in pat.finditer(text):
        doc, name, sig = m.group(1), m.group(2), m.group(3)
        hit = LIMIT.search(CODE.sub("", doc))
        if hit and HYP.search(sig) and not LEAN_LABEL.search(doc):
            line = text.count("\n", 0, m.start()) + 1
            fails.append((str(path), line, line, f"{name}: '{hit.group(0)}' over an h-hypothesis, no grade"))
    return fails


def scope_files(root):
    md = sorted({p for g in MD_SCOPE for p in root.glob(g)})
    return md, [root / p for p in LEAN_SCOPE if (root / p).exists()]


def run_check(root):
    return run_check_with_index(root, theorem_index(root))


def _staged_renames(root):
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


def _module_path(path):
    """``src/gpu/simt/mod.rs`` -> ``crate::gpu::simt``; None for a non-``src`` Rust path."""
    if not (path.startswith("src/") and path.endswith(".rs")):
        return None
    rel = path[4:-3]
    if rel == "mod":
        rel = ""
    elif rel.endswith("/mod"):
        rel = rel[:-4]
    return "crate::" + rel.replace("/", "::")


def _substitutions(renames):
    """Ordered ``old -> new`` path/module token pairs from the staged rename map.

    Kept in step with ``tools/hooks/perf_gate_controls.py``: exact path, ``crate::`` module path
    of a ``.rs`` rename, and a directory prefix whose every rename agrees on the destination.
    """
    pairs = []
    for new, old in renames.items():
        pairs.append((old, new))
        old_mod, new_mod = _module_path(old), _module_path(new)
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


def _normaliser(table):
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


HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")


def run_staged(root):
    normalise = _normaliser(_substitutions(_staged_renames(root)))
    out = subprocess.run(["git", "-C", str(root), "diff", "--cached", "-M", "-U0", "--", "docs/*.md",
                          "book/*.md", "docs/**/*.md", "book/**/*.md"],
                         capture_output=True, text=True, check=True).stdout
    added, path, line_no = {}, None, 0
    removed, added_lines = [], []

    def flush():
        """Record the hunk's added lines unless every one pairs with an equal removed line."""
        path_only = len(removed) == len(added_lines) and all(
            normalise(old) == normalise(new) for old, (_, new) in zip(removed, added_lines)
        )
        if not path_only:
            for number, _ in added_lines:
                added.setdefault(path, set()).add(number)
        removed.clear()
        added_lines.clear()

    for line in out.split("\n"):
        if line.startswith("+++ "):
            flush()
            path = None if line.endswith("/dev/null") else line[6:]
            continue
        if line.startswith("--- "):
            continue
        m = HUNK.match(line)
        if m:
            flush()
            line_no = int(m.group(1))
            continue
        if path is None:
            continue
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added_lines.append((line_no, line[1:]))
            line_no += 1
        elif line.startswith(" "):
            line_no += 1
    flush()
    idx, beaten = theorem_index(root), floor_literals(root)
    fails = []
    for path, lines in added.items():
        text = subprocess.run(["git", "-C", str(root), "show", f":{path}"], capture_output=True, text=True,
                              check=True).stdout
        for f in check_md(path, text, idx, beaten):
            if any(f[1] <= n <= f[2] for n in lines):
                fails.append(f)
    return fails


def report(fails):
    for path, a, b, why in fails:
        print(f"limit-check: {path}:{a}-{b}: {why}")
    print(f"limit-check: {len(fails)} failure(s)")
    return 1 if fails else 0


FIXTURES = [
    ("bad: bare floor", "## Results\n\nAt 1.18 b/w the floor is 16.9x Q4.\n", 1),
    ("bad: cannot reach", "- K-FAC allocation cannot reach 1.18 b/w.\n", 1),
    ("bad: unprecedented", "This result is unprecedented.\n", 1),
    ("bad: proved over h-hypothesis", "Lower bound, `proved` by `cond_thm`.\n", 1),
    ("bad: beaten floor without refuted", "- the `model` floor `0.3887` nats\n", 1),
    ("good: inline label", "- the floor is 16.9x Q4 (`model`).\n", 0),
    ("good: heading label", "## Floor (`anchor`)\n\nThe floor is 16.9x Q4.\n", 0),
    ("good: proved unconditional", "Lower bound, `proved` by `free_thm`.\n", 0),
    ("good: inline code only", "Run `kfac_floor.py` and `converse.py`.\n", 0),
    ("good: beaten floor refuted", "- the `model` floor `0.3887` nats, `refuted` by 0.35064\n", 0),
]


STAGED_BASE = "## Results\n\n- the floor `src/simt/x.rs` cannot reach 1.18 b/w.\n"
STAGED_RECEIPT = {
    "variants": {
        "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp-nb": {
            "kl_insample_stats": {"mean": 0.35064}
        }
    }
}


def _staged_fixture(workdir, body, move_src=True, rename_doc=False):
    """Stage a doc rewrite (and its source rename) in a throwaway repo; return the gate's exit."""
    repo = workdir / "staged"
    if repo.exists():
        shutil.rmtree(repo)
    (repo / "src" / "simt").mkdir(parents=True)
    (repo / "src" / "simt" / "x.rs").write_text("pub struct X;\n")
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "note.md").write_text(STAGED_BASE)
    receipt = repo / "docs" / "evidence" / "receipts" / "onebit" / "t3-nb-3seed.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(STAGED_RECEIPT))
    for args in (("init", "-q"), ("config", "user.email", "limit@example.invalid"),
                 ("config", "user.name", "limit control"), ("add", "-A"), ("commit", "-q", "-m", "base")):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    if rename_doc:
        subprocess.run(["git", "-C", str(repo), "mv", "docs/note.md", "docs/moved.md"],
                       check=True, capture_output=True)
    else:
        if move_src:
            (repo / "src" / "gpu" / "simt").mkdir(parents=True)
            subprocess.run(["git", "-C", str(repo), "mv", "src/simt/x.rs", "src/gpu/simt/x.rs"],
                           check=True, capture_output=True)
        (repo / "docs" / "note.md").write_text(body)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    return 1 if run_staged(repo) else 0


def self_test():
    idx = {"cond_thm": False, "free_thm": True}
    beaten = [("0.3887", "fixture.json", 0.35064)]
    bad = 0
    for name, text, want in FIXTURES:
        got = 1 if check_md("fixture.md", text, idx, beaten) else 0
        ok = got == want
        bad += not ok
        print(f"limit-check self-test: {'ok  ' if ok else 'FAIL'} {name} -> {'red' if got else 'green'}")
    lean_bad = "/-- the converse bound -/\ntheorem t (hslb : x ≤ y) : x ≤ y := hslb\n"
    lean_ok = "/-- the converse bound, `anchor` for the Shannon side -/\ntheorem t (hslb : x ≤ y) : x ≤ y := hslb\n"
    for name, text, want in (("bad: lean ungraded", lean_bad, 1), ("good: lean anchor", lean_ok, 0)):
        got = 1 if check_lean("fixture.lean", text) else 0
        bad += got != want
        print(f"limit-check self-test: {'ok  ' if got == want else 'FAIL'} {name} -> {'red' if got else 'green'}")
    workdir = Path(tempfile.mkdtemp(dir="/home/socol/oc-tmp", prefix="limit-check-staged-"))
    try:
        staged = [
            ("good: path-only rewrite", dict(body=STAGED_BASE.replace("src/simt/", "src/gpu/simt/")), 0),
            ("bad: path rewrite plus one changed token",
             dict(body=STAGED_BASE.replace("src/simt/", "src/gpu/simt/").replace("1.18", "1.19")), 1),
            ("good: pure rename", dict(body=STAGED_BASE, move_src=False, rename_doc=True), 0),
        ]
        for name, kwargs, want in staged:
            got = _staged_fixture(workdir, **kwargs)
            ok = got == want
            bad += not ok
            print(f"limit-check self-test: {'ok  ' if ok else 'FAIL'} {name} -> {'red' if got else 'green'}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    total = len(FIXTURES) + 2 + len(staged)
    print(f"limit-check self-test: {total - bad}/{total}")
    return 1 if bad else 0


def controls():
    """Mutate copies of the live scope; each mutation must turn the check red."""
    base = run_check(REPO)
    print(f"limit-check controls: baseline {'green' if not base else 'RED'}")
    muts = [
        ("strip the W15c labels from the K-FAC rate claim", "docs/evidence/onebit-w15.md",
         lambda s: s.replace("side (`anchor`) and the K-FAC floor (`model`)", "side and the K-FAC floor", 1)
         .replace(" (`model`: K-FAC quadratic KL,\n  Gaussian rate).", ".", 1)),
        ("strip `refuted` from the beaten 0.3887 floor", "docs/evidence/onebit-w15.md",
         lambda s: s.replace("`refuted`", "refuted")),
        ("restore the premise-free waterfill_floor docstring", "proof/corpus/Perf/CostCorrespondence.lean",
         lambda s: re.sub(r"The water level is the floor, under the exponential distortion model\.\*\* If .*?then at level `θ`\n",
                          "The water level is the floor.** At level `θ` ", s, count=1, flags=re.S)),
        ("add an ungraded impossibility", "docs/evidence/onebit-w13.md",
         lambda s: s + "\n\nQ4 parity at 1.18 b/w is impossible.\n"),
    ]
    bad = bool(base)
    for name, rel, f in muts:
        with tempfile.TemporaryDirectory(dir="/home/socol/oc-tmp") as t:
            t = Path(t)
            for p in [*scope_files(REPO)[0], *scope_files(REPO)[1]]:
                q = t / p.relative_to(REPO)
                q.parent.mkdir(parents=True, exist_ok=True)
                q.write_text(p.read_text())
            for _, rec, _ in FLOORS:
                (t / rec).parent.mkdir(parents=True, exist_ok=True)
                (t / rec).write_text((REPO / rec).read_text())
            for d in ("proof/corpus", "proof/canonical"):
                (t / d).mkdir(parents=True, exist_ok=True)
            tgt = t / rel
            before = tgt.read_text()
            after = f(before)
            # theorem signatures come from the live tree; the copies carry only the scoped files
            fails = run_check_with_index(t, theorem_index(REPO))
            assert not fails, f"copy baseline red: {fails[:2]}"
            tgt.write_text(after)
            fails = run_check_with_index(t, theorem_index(REPO))
            red = bool(fails) and after != before
            bad += not red
            print(f"limit-check controls: {'ok  ' if red else 'FAIL'} {name} -> {'red' if red else 'green'}"
                  + (f" ({fails[0][0]}:{fails[0][1]})" if fails else ""))
            tgt.write_text(before)
            restored = not run_check_with_index(t, theorem_index(REPO))
            bad += not restored
            print(f"limit-check controls: {'ok  ' if restored else 'FAIL'} restore -> {'green' if restored else 'red'}")
    return 1 if bad else 0


def run_check_with_index(root, idx):
    beaten = floor_literals(root)
    md, lean = scope_files(root)
    fails = []
    for p in md:
        fails += check_md(p.relative_to(root), p.read_text(), idx, beaten)
    for p in lean:
        fails += check_lean(p.relative_to(root), p.read_text())
    return fails


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--staged", action="store_true")
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--controls", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.controls:
        return controls()
    return report(run_staged(REPO) if a.staged else run_check(REPO))


if __name__ == "__main__":
    sys.exit(main())
