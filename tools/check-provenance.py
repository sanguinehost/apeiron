#!/usr/bin/env python3
"""Provenance gate: every ported file and every file of unstated origin is named, and the lists agree.

  check-provenance.py                 structural check (no external tools)
  check-provenance.py --similarity    LLAMA_CPP=<llama.cpp checkout with both pinned revisions fetched>;
                                      recompute each ported file's token 5-gram containment and compare
  check-provenance.py --recompile     glslc on PATH; every .spv must equal glslc's output byte for byte
  check-provenance.py --self-test     seeded defects, each of which must fail with its own diagnostic

Structural check: the files `NOTICE` lists as ported equal `[[ported]]` in `tools/provenance.toml`; each
exists and names at least one upstream path; every file under `[coverage]` matches an `[[origin]]` glob
and every glob matches a file.
"""
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?|\S")
COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*|(?m:^[ \t]*#[^\n]*)", re.S)
N = 5


def notice_files(text):
    return set(re.findall(r"(?m)^  (\S+\.(?:rs|c|h|cpp|py))\s", text))


def covered_files(root, globs):
    out = set()
    for g in globs:
        out |= {str(p.relative_to(root)) for p in root.glob(g) if p.is_file()}
    return out


def structural(root, notice, cfg):
    """Return the list of failures."""
    bad = []
    ported = {e["file"]: e for e in cfg.get("ported", [])}
    listed = notice_files(notice)
    for f in sorted(listed - ported.keys()):
        bad.append(f"NOTICE lists {f}, tools/provenance.toml has no [[ported]] for it")
    for f in sorted(ported.keys() - listed):
        bad.append(f"tools/provenance.toml ports {f}, NOTICE does not list it")
    for f, e in sorted(ported.items()):
        if not (root / f).is_file():
            bad.append(f"ported file {f} does not exist")
        if not e.get("upstream"):
            bad.append(f"ported file {f} names no upstream path")
        for u in e.get("upstream", []):
            if u["rev"] not in cfg["pins"]:
                bad.append(f"{f}: upstream rev {u['rev']!r} is not in [pins]")
    origins = cfg.get("origin", [])
    files = covered_files(root, cfg["coverage"]["globs"])
    for f in sorted(files):
        if not any(fnmatch.fnmatch(f, o["glob"]) for o in origins):
            bad.append(f"{f} has no [[origin]] entry")
    for o in origins:
        if not any(fnmatch.fnmatch(f, o["glob"]) for f in files):
            bad.append(f"[[origin]] glob {o['glob']} matches no file in the coverage set")
        if not o.get("text", "").strip():
            bad.append(f"[[origin]] glob {o['glob']} has no text")
        if "glslc" in o:
            for f in files:
                if fnmatch.fnmatch(f, o["glob"]) and not (root / source_of(o, f)).is_file():
                    bad.append(f"{f}: source {source_of(o, f)} does not exist")
    return bad


def source_of(origin, path):
    return origin.get("source") or str(Path(path).with_suffix(".comp"))


def shingles(text):
    text = COMMENT.sub(" ", text)
    toks = [t.lower().replace("_", "") if t[0].isalpha() or t[0] == "_" else t for t in TOKEN.findall(text)]
    return {tuple(toks[i:i + N]) for i in range(len(toks) - N + 1)}


def containment(rust, upstream_texts):
    mine = shingles(rust)
    theirs = set()
    for t in upstream_texts:
        theirs |= shingles(t)
    return round(len(mine & theirs) / len(mine), 3) if mine else 0.0


def similarity(root, cfg, llama):
    bad, rows = [], []
    for e in cfg["ported"]:
        texts = []
        for u in e["upstream"]:
            rev = cfg["pins"][u["rev"]]
            r = subprocess.run(["git", "-C", llama, "show", f"{rev}:{u['path']}"], capture_output=True, text=True)
            if r.returncode:
                bad.append(f"{e['file']}: git show {rev}:{u['path']} failed: {r.stderr.strip()}")
                continue
            texts.append(r.stdout)
        got = containment((root / e["file"]).read_text(), texts)
        rows.append((e["file"], got, e["containment"]))
        if got != e["containment"]:
            bad.append(f"{e['file']}: containment {got}, tools/provenance.toml records {e['containment']}")
    return bad, rows


def recompile(root, cfg):
    bad = []
    files = covered_files(root, cfg["coverage"]["globs"])
    with tempfile.TemporaryDirectory() as tmp:
        for f in sorted(files):
            for o in cfg["origin"]:
                if "glslc" in o and fnmatch.fnmatch(f, o["glob"]):
                    out = Path(tmp) / Path(f).name
                    r = subprocess.run(["glslc", *o["glslc"], str(root / source_of(o, f)), "-o", str(out)],
                                       capture_output=True, text=True)
                    if r.returncode:
                        bad.append(f"{f}: glslc failed: {r.stderr.strip()}")
                    elif out.read_bytes() != (root / f).read_bytes():
                        bad.append(f"{f}: not the glslc output of {source_of(o, f)}")
                    break
    return bad


def load():
    return (ROOT / "NOTICE").read_text(), tomllib.loads((ROOT / "tools/provenance.toml").read_text())


def self_test():
    notice, cfg = load()
    expect = []

    def control(name, bad, needle):
        ok = any(needle in b for b in bad)
        expect.append(ok)
        print(f"{'ok  ' if ok else 'FAIL'} control {name}: {bad[0] if bad else 'no failure'}")

    base = structural(ROOT, notice, cfg)
    expect.append(not base)
    print(f"{'ok  ' if not base else 'FAIL'} baseline: {len(base)} failures")
    f = cfg["ported"][0]["file"]
    control("file removed from NOTICE", structural(ROOT, notice.replace(f"  {f}", "  removed/x.rs"), cfg), f"ports {f}")
    cut = {**cfg, "ported": cfg["ported"][1:]}
    control("file removed from the toml", structural(ROOT, notice, cut), "has no [[ported]]")
    cut = {**cfg, "origin": [o for o in cfg["origin"] if o["glob"] != "fold-abi/*.md"]}
    control("origin removed", structural(ROOT, notice, cut), "fold-abi/SMHASHER.md has no [[origin]]")
    cut = {**cfg, "origin": cfg["origin"] + [{"glob": "nowhere/*", "text": "x"}]}
    control("origin matches nothing", structural(ROOT, notice, cut), "matches no file")
    same = "fn a() { let x = foo_bar(1, 2, 3) + baz; }"
    expect += [containment(same, [same]) == 1.0, containment(same, ["int q;"]) == 0.0]
    print(f"{'ok  ' if all(expect[-2:]) else 'FAIL'} control containment: identical 1.0, disjoint 0.0")
    if shutil.which("glslc"):
        with tempfile.TemporaryDirectory() as tmp:
            for g in covered_files(ROOT, cfg["coverage"]["globs"]):
                (Path(tmp) / g).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / g, Path(tmp) / g)
            clean = recompile(Path(tmp), cfg)
            expect.append(not clean)
            print(f"{'ok  ' if not clean else 'FAIL'} recompile baseline: {len(clean)} failures")
            victim = Path(tmp) / "src/spirv/splitk_reduce.spv"
            b = victim.read_bytes()
            victim.write_bytes(b[:-1] + bytes([b[-1] ^ 0xFF]))
            control("spv byte changed", recompile(Path(tmp), cfg), "splitk_reduce.spv: not the glslc output")
    return all(expect)


def main():
    a = sys.argv[1:]
    if a == ["--self-test"]:
        sys.exit(0 if self_test() else 1)
    notice, cfg = load()
    bad = structural(ROOT, notice, cfg)
    if a == ["--similarity"]:
        llama = os.environ.get("LLAMA_CPP")
        if not llama:
            sys.exit("LLAMA_CPP is not set")
        more, rows = similarity(ROOT, cfg, llama)
        bad += more
        for f, got, rec in rows:
            print(f"{got:5.3f} (recorded {rec:5.3f})  {f}")
    elif a == ["--recompile"]:
        bad += recompile(ROOT, cfg)
    elif a:
        sys.exit(__doc__)
    for b in bad:
        print("FAIL", b)
    print(f"provenance: {len(cfg['ported'])} ported, {len(cfg['origin'])} origin entries, {len(bad)} failures")
    sys.exit(1 if bad else 0)


main()
