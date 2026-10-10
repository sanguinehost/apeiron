#!/usr/bin/env python3
"""Public cut: internal tree -> publishable clone, with history rewritten.

  release/cut.py build [--rev REV] OUT      clone REV (default HEAD) of this repository into OUT,
                                            apply release/manifest.toml, add the substitutions
  release/cut.py check [--full] OUT         scan OUT against the manifest (exit 0 = clean);
                                            --full also builds and tests it
  release/cut.py self-test                  build a clone, mutate it, and require `check` to name
                                            each mutation

The manifest is the only configuration. `build` never touches the source repository.
"""
import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tempfile
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
MANIFEST = os.path.join(HERE, "manifest.toml")


def load():
    with open(MANIFEST, "rb") as f:
        return tomllib.load(f)


def run(cmd, cwd=None, check=True, inp=None, env=None):
    r = subprocess.run(cmd, cwd=cwd, input=inp, capture_output=True, env=env)
    if check and r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace"))
        raise SystemExit(f"cut: failed: {' '.join(cmd)}")
    return r


def git_show(rev, path, cwd=SRC):
    return run(["git", "show", f"{rev}:{path}"], cwd=cwd).stdout


def unpublished_rules(m):
    """One rewrite per stripped prefix: a shipped mention keeps the name and says it is not shipped."""
    skip = set(m["expect"].get("dangling_ignore", []))
    subs = list(m["substitute"])
    rules = []
    for p in m["strip"]:
        if p in skip or p.startswith("docs/") or any(c in p for c in "*?["):
            continue
        if any(d == p or (p.endswith("/") and d.startswith(p)) for d in subs):
            continue
        tail = r"(?![\w/\-]|\.\w| \(unpublished\))"
        if p.endswith("/"):
            rx = r"(?<![\w/.\-])(" + re.escape(p) + r"[\w./\-]*[\w])" + tail
        else:
            rx = r"(?<![\w/.\-])(" + re.escape(p) + r")" + tail
        rules.append("regex:" + rx + r"==>\1 (unpublished)")
    return rules


def build(out, rev):
    m = load()
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"cut: {out} exists and is not empty")
    full = run(["git", "rev-parse", rev], cwd=SRC).stdout.decode().strip()
    run(["git", "clone", "-q", "--no-local", "--single-branch", "--branch", "main", SRC, out])
    run(["git", "checkout", "-q", "--detach", full], cwd=out)
    run(["git", "branch", "-q", "-f", "main", full], cwd=out)
    run(["git", "checkout", "-q", "main"], cwd=out)
    subs = {}
    for dst, srcpath in m["substitute"].items():
        subs[dst] = git_show(full, srcpath)

    with tempfile.TemporaryDirectory() as td:
        text = os.path.join(td, "text.txt")
        msg = os.path.join(td, "msg.txt")
        with open(text, "w") as f:
            f.write("\n".join(m["text_map"] + unpublished_rules(m)) + "\n")
        with open(msg, "w") as f:
            f.write("\n".join(m["message_map"] + m["text_map"] + unpublished_rules(m)) + "\n")
        cmd = ["git", "filter-repo", "--force", "--invert-paths"]
        for p in m["strip"]:
            if p.endswith("/"):
                cmd += ["--path", p]
            elif any(c in p for c in "*?["):
                cmd += ["--path-glob", p]
            else:
                cmd += ["--path", p]
        cmd += ["--replace-text", text, "--replace-message", msg]
        a = m["authors"]
        mailmap = os.path.join(td, "mailmap")
        with open(mailmap, "w") as f:
            for alias in a["aliases"]:
                f.write(f"{a['name']} <{a['email']}> {alias}\n")
        cmd += ["--mailmap", mailmap]
        run(cmd, cwd=out)

    remap_revisions(out, m)
    for dst, data in subs.items():
        p = os.path.join(out, dst)
        os.makedirs(os.path.dirname(p) or out, exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        if dst.startswith(".githooks/") or dst.endswith(".sh"):
            os.chmod(p, 0o755)
    run(["git", "add", "-A"], cwd=out)
    body = "public: contributor guide, hooks and installer for the public tree\n\n" + \
           "Adds " + ", ".join(sorted(subs)) + "."
    ident = run(["git", "log", "-1", "--format=%an%n%ae", full], cwd=SRC).stdout.decode().split("\n")
    env = dict(os.environ, GIT_AUTHOR_NAME=ident[0], GIT_AUTHOR_EMAIL=ident[1],
               GIT_COMMITTER_NAME=ident[0], GIT_COMMITTER_EMAIL=ident[1])
    run(["git", "commit", "-q", "-s", "-m", body], cwd=out, env=env)
    print(f"cut: built {out} from {full[:12]}")


def remap_revisions(out, m):
    """Rewrite abbreviated commit ids in `remap` paths to the ids of the rewritten history.

    The filter changes every commit id, so a document that pins evidence to a revision would
    otherwise cite one that does not resolve in the cut. An id whose commit left the cut (it only
    touched stripped paths) is left alone."""
    cm = os.path.join(out, ".git", "filter-repo", "commit-map")
    if not os.path.exists(cm):
        return
    pairs = {}
    with open(cm) as f:
        next(f)
        for line in f:
            old, new = line.split()
            if set(new) != {"0"}:
                pairs[old] = new
    pat = re.compile(r"\b[0-9a-f]{7,40}\b")

    def sub(mt):
        tok = mt.group(0)
        hit = [o for o in pairs if o.startswith(tok)]
        return pairs[hit[0]][:len(tok)] if len(hit) == 1 else tok

    for f in tree_files(out):
        if not any(f == r or (r.endswith("/") and f.startswith(r)) for r in m["expect"].get("remap", [])):
            continue
        path = os.path.join(out, f)
        try:
            s = open(path, encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        r = pat.sub(sub, s)
        if r != s:
            open(path, "w", encoding="utf-8").write(r)


def tree_files(out):
    r = run(["git", "ls-files", "-z"], cwd=out)
    return [p for p in r.stdout.decode().split("\0") if p]


def strip_hits(files, strip):
    hits = []
    for f in files:
        for p in strip:
            if p.endswith("/"):
                if f.startswith(p):
                    hits.append((f, p))
            elif any(c in p for c in "*?["):
                if fnmatch.fnmatch(f, p) or fnmatch.fnmatch("/" + f, "*" + p.lstrip("*")):
                    hits.append((f, p))
            elif f == p:
                hits.append((f, p))
    return hits


def objects(out):
    """Yield (sha, type, bytes) for every object of the clone."""
    p = subprocess.Popen(["git", "cat-file", "--batch-all-objects", "--batch"], cwd=out,
                         stdout=subprocess.PIPE)
    rd = p.stdout
    while True:
        head = rd.readline()
        if not head:
            break
        sha, typ, size = head.decode().split()
        data = rd.read(int(size))
        rd.read(1)
        yield sha, typ, data
    p.wait()


LAST = []


def check(out, full):
    m = load()
    bad = []
    files = tree_files(out)
    fileset = set(files)

    for f, p in strip_hits(files, m["strip"]):
        if f in m["substitute"]:
            continue
        bad.append(f"stripped path present: {f} (matches {p})")
    for f in files:
        if f.startswith("docs/") and not any(fnmatch.fnmatch(f, g) for g in m["expect"]["docs_allow"]):
            bad.append(f"docs file outside docs_allow: {f}")
    for f in files:
        if any(f == d or (d.endswith("/") and f.startswith(d)) for d in m["expect"]["public_docs"]):
            with open(os.path.join(out, f), "rb") as fh:
                for i, ln in enumerate(fh.read().decode(errors="replace").splitlines(), 1):
                    if "unpublished" in ln:
                        bad.append(f"rewritten pointer in public document {f}:{i}: {ln.strip()[:90]}")
    for p in m["expect"]["present"]:
        if p not in fileset:
            bad.append(f"expected path missing: {p}")
    for dst in m["substitute"]:
        if dst not in fileset:
            bad.append(f"substitution missing: {dst}")

    pats = [(s, re.compile(s.encode(), re.M)) for s in m["forbidden"]]
    want = f"{m['authors']['name']} <{m['authors']['email']}>"
    seen = 0
    for sha, typ, data in objects(out):
        if typ == "commit":
            body = data.split(b"\n\n", 1)[1] if b"\n\n" in data else b""
            seen += 1
            head = data.split(b"\n\n", 1)[0].decode(errors="replace")
            for ln in head.splitlines():
                if ln.startswith(("author ", "committer ")):
                    who = ln.split(" ", 1)[1].rsplit(" ", 2)[0]
                    if who != want:
                        bad.append(f"commit {sha[:12]} {ln.split(' ', 1)[0]} is {who}, not {want}")
            for s, rx in pats:
                if rx.search(body):
                    bad.append(f"commit {sha[:12]} message matches /{s}/")
        elif typ == "blob":
            for s, rx in pats:
                mo = rx.search(data)
                if mo:
                    line = data.count(b"\n", 0, mo.start()) + 1
                    bad.append(f"blob {sha[:12]} line {line} matches /{s}/")
    # Blob hits name a sha; resolve to a path in HEAD or history.
    if bad:
        bad = resolve_blobs(out, bad)

    # A shipped file that names a stripped path that no longer resolves is a dangling reference.
    ign = set(m["expect"]["dangling_ignore"])
    dirs = [p for p in m["strip"] if p.endswith("/") and p not in ign]
    exact = [p for p in m["strip"] if not p.endswith("/") and p not in ign and not any(c in p for c in "*?[")]
    names = [re.escape(p) for p in dirs + exact]
    rx = re.compile(r"(?<![\w/.\-])(" + "|".join(names) + r")[\w./\-]*")
    dirset = set()
    for f in files:
        parts = f.split("/")
        for i in range(1, len(parts)):
            dirset.add("/".join(parts[:i]) + "/")
    for f in files:
        if f == ".gitignore":
            continue
        try:
            with open(os.path.join(out, f), "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        if b"\0" in data[:4096]:
            continue
        txt = data.decode(errors="replace")
        for mo in rx.finditer(txt):
            tok = mo.group(0).rstrip(".,;:)`'\"")
            if tok in fileset or tok in dirset or tok.rstrip("/") + "/" in dirset:
                continue
            if txt.startswith(" (unpublished)", mo.start() + len(tok)):
                continue
            bad.append(f"dangling reference in {f}:{txt.count(chr(10), 0, mo.start()) + 1}: {tok}")

    if full and not bad:
        bad += full_gate(out)
    global LAST
    LAST = list(dict.fromkeys(bad))
    if bad:
        for b in LAST:
            print("cut: FAIL", b)
        print(f"cut: check FAILED ({len(set(bad))} finding(s))")
        return 1
    print(f"cut: check PASS ({len(files)} files, {seen} commits, "
          f"{len(pats)} forbidden patterns, {len(dirs) + len(exact)} stripped prefixes)")
    return 0


def resolve_blobs(out, bad):
    shas = {b.split()[1] for b in bad if b.startswith("blob ")}
    if not shas:
        return bad
    r = run(["git", "rev-list", "--objects", "--all"], cwd=out).stdout.decode().splitlines()
    where = {}
    for ln in r:
        parts = ln.split(" ", 1)
        if len(parts) == 2:
            where.setdefault(parts[0][:12], parts[1])
    return [re.sub(r"^blob (\w+)", lambda mo: f"blob {mo.group(1)} ({where.get(mo.group(1), '?')})", b)
            for b in bad]


def full_gate(out):
    """cargo and book gates on the clone, each bounded (apeiron AGENTS.md section 6)."""
    bad = []
    bound = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=16G", "-p", "MemorySwapMax=0",
             "timeout", "580"]
    steps = [
        ("check-claims self-test", ["python3", "-B", "tools/check-claims.py", "--self-test"]),
        ("check-claims", ["python3", "-B", "tools/check-claims.py"]),
        ("cargo fmt", ["cargo", "fmt", "--all", "--", "--check"]),
        ("cargo test", ["cargo", "test", "--locked", "--lib", "--test", "table"]),
        ("book", ["bash", "tools/book-test.sh"]),
    ]
    env = dict(os.environ)
    env["PATH"] = "/home/socol/oc-tmp/mdbook/bin:" + env["PATH"]
    for name, cmd in steps:
        r = subprocess.run(bound + cmd, cwd=out, capture_output=True, env=env)
        print(f"cut: {name}: exit {r.returncode}")
        if r.returncode != 0:
            tail = (r.stdout + r.stderr).decode(errors="replace").splitlines()[-8:]
            bad.append(f"{name} exit {r.returncode}: " + " | ".join(tail))
    return bad


def self_test():
    ok = True
    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/oc-tmp")) as td:
        out = os.path.join(td, "clone")
        build(out, "HEAD")
        print("-- baseline (must pass)")
        if check(out, False) != 0:
            print("cut: self-test FAIL baseline")
            return 1
        m = load()
        controls = [
            ("stripped path", "docs/plan.md", b"# plan\n", "stripped path present: docs/plan.md"),
            ("home path", "src/zz_control.rs", b"// /home/socol/x\n", "(src/zz_control.rs)"),
            ("row id", "src/zz_control2.rs", b"// rel-apeiron-cut-GTmRyI2IZuimzE0Sn87cNf\n", "(src/zz_control2.rs)"),
            ("dangling", "src/zz_control3.rs", b"// see tools/rooting/check.py\n",
             "dangling reference in src/zz_control3.rs"),
        ]
        for name, path, data, tag in controls:
            p = os.path.join(out, path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
            run(["git", "add", "-f", path], cwd=out)
            run(["git", "commit", "-q", "-m", f"control {name}"], cwd=out)
            print(f"-- control: {name} (must fail)")
            rc = check(out, False)
            if rc == 0:
                print(f"cut: self-test FAIL: control '{name}' passed")
                ok = False
            elif not any(tag in b for b in LAST):
                print(f"cut: self-test FAIL: control '{name}' failed without naming {tag!r}")
                ok = False
            run(["git", "rm", "-q", "-f", path], cwd=out)
            run(["git", "commit", "-q", "-m", f"restore {name}"], cwd=out)
        # An author control cannot be restored either.
        env = dict(os.environ, GIT_AUTHOR_NAME="agent", GIT_AUTHOR_EMAIL="agent@local")
        run(["git", "commit", "-q", "--allow-empty", "-m", "x"], cwd=out, env=env)
        print("-- control: author identity (must fail)")
        if check(out, False) == 0:
            print("cut: self-test FAIL: author control passed")
            ok = False
        elif not any(b.startswith("commit ") and "agent <agent@local>" in b for b in LAST):
            print("cut: self-test FAIL: author control failed without naming the author")
            ok = False
        # A message control cannot be restored by a later commit: history keeps it.
        run(["git", "commit", "-q", "--allow-empty", "-m", "x\n\nboard: rel-x"], cwd=out)
        print("-- control: board trailer in a message (must fail)")
        if check(out, False) == 0:
            print("cut: self-test FAIL: message control passed")
            ok = False
        elif not any(b.startswith("commit ") and "board" in b for b in LAST):
            print("cut: self-test FAIL: message control failed without naming a commit message")
            ok = False
    print("cut: self-test", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--rev", default="HEAD")
    b.add_argument("out")
    c = sub.add_parser("check")
    c.add_argument("--full", action="store_true")
    c.add_argument("out")
    sub.add_parser("self-test")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.rev)
        return 0
    if a.cmd == "check":
        return check(a.out, a.full)
    return self_test()


if __name__ == "__main__":
    sys.exit(main())
