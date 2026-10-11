#!/usr/bin/env python3
"""The derivation gate: a Claude Code PreToolUse hook that refuses runs taken in place of thinking.

Owner direction, 2026-10-11, after the 2026-10-10 sweeps. The harness-level version is the intent
lane's derivation-gate stage; this hook covers P1 and P3 in this repository until that stage lands.

P1  A Bash command in the run class (.intent/runs.toml (unpublished)) needs PREDICTION=<id> naming an open
    prediction. A prediction admits one command; identical repeats of that command (averaging) are
    admitted up to MAX_REPEATS. A command whose skeleton (numbers replaced) matches an earlier run
    under a different prediction with the same basis is a sweep and is refused.
P3  No foreground waiting: a foreground Bash call with timeout > FOREGROUND_MAX_MS is refused; a
    run-class call must be background or declare timeout <= FOREGROUND_MAX_MS.
    Ledger integrity: the ledgers, the run class, this file and .claude/settings*.json are not
    writable through Write/Edit or by any Bash command other than this script's own CLI.

CLI (the only write path into the ledger):
    derivation_gate.py predict --q SLUG --question TEXT --predicted TEXT --basis BASIS
                               --falsifier TEXT --wall-s N
        BASIS is `path:line` resolving to an existing line (repo-relative, ~/Workspace-relative or
        absolute), optionally followed by ` -- note`, or `unproved: <derivation, <= 5 lines>`.
        Refused while MAX_OPEN predictions have no result.
    derivation_gate.py result ID confirmed|refuted|withdrawn --measured TEXT [--receipt PATH]
    derivation_gate.py status
    derivation_gate.py self-test

Fails closed: an internal error in hook mode refuses the call. Trust boundary: the ledger records
self-reported outcomes; an obfuscated command that writes the ledger without naming it is not seen.

Cost: per hook call O(P + R) over the prediction and run records (one pass over each JSONL file).
"""
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOREGROUND_MAX_MS = 60_000
MAX_REPEATS = 10
MAX_OPEN = 3
PROTECTED = ("predictions.jsonl", "runs.jsonl", "runs.toml", "derivation_gate", ".claude/settings")
CLI_RE = re.compile(r"^\s*python3\s+(?:\S*/)?\.claude/hooks/derivation_gate\.py\s+(predict|result|status|self-test)\b")
PRED_RE = re.compile(r"\bPREDICTION=([A-Za-z0-9_.-]+)")
NUM_RE = re.compile(r"(?<![A-Za-z_])[0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?")


class Ledger:
    def __init__(self, root):
        self.dir = root / ".intent"
        self.preds_path = self.dir / "predictions.jsonl"
        self.runs_path = self.dir / "runs.jsonl"
        self.root = root

    def _read(self, path):
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def _append(self, path, rec):
        self.dir.mkdir(exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def predictions(self):
        """id -> merged record (prediction plus its result, if any)."""
        out = {}
        for rec in self._read(self.preds_path):
            if rec["kind"] == "prediction":
                out[rec["id"]] = dict(rec)
            elif rec["kind"] == "result" and rec["id"] in out:
                out[rec["id"]]["result"] = rec
        return out

    def runs(self):
        return self._read(self.runs_path)

    def run_class(self):
        cfg = tomllib.loads((self.dir / "runs.toml").read_text())
        return [re.compile(p) for p in cfg["run_class"]]


def skeleton(cmd):
    return NUM_RE.sub("#", PRED_RE.sub("", " ".join(cmd.split())))


def basis_ok(basis, root):
    if basis.startswith("unproved:"):
        body = basis[len("unproved:"):].strip()
        lines = body.splitlines()
        return (bool(body) and len(lines) <= 5), "unproved basis must be 1-5 non-empty lines"
    ref = basis.split(" -- ", 1)[0].strip()
    m = re.fullmatch(r"(.+):([0-9]+)", ref)
    if not m:
        return False, "basis is neither `path:line` nor `unproved: ...`"
    path, line = m.group(1), int(m.group(2))
    for base in (root, Path.home() / "Workspace", Path("/")):
        p = (base / os.path.expanduser(path)).resolve()
        if p.is_file():
            n = sum(1 for _ in p.open(errors="replace"))
            if 1 <= line <= n:
                return True, ""
            return False, f"{path} has {n} lines, basis names line {line}"
    return False, f"basis path {path} does not resolve"


def decide_bash(inp, led):
    """Returns None to admit, or the refusal text."""
    cmd = inp.get("command", "")
    bg = bool(inp.get("run_in_background"))
    timeout = inp.get("timeout")
    if any(p in cmd for p in PROTECTED) and not cli_only(cmd):
        return "derivation gate: the command names a protected ledger/gate file; only `python3 .claude/hooks/derivation_gate.py predict|result|status` may touch it"
    if not bg and timeout is not None and timeout > FOREGROUND_MAX_MS:
        return f"derivation gate P3: foreground timeout {timeout} ms > {FOREGROUND_MAX_MS} ms; run it in the background and keep deriving"
    if not any(p.search(cmd) for p in led.run_class()):
        return None
    if not bg and (timeout is None or timeout > FOREGROUND_MAX_MS):
        return f"derivation gate P3: a run-class command must run in the background or declare timeout <= {FOREGROUND_MAX_MS} ms"
    m = PRED_RE.search(cmd)
    if not m:
        return "derivation gate P1: run-class command without PREDICTION=<id>; write the prediction first (`derivation_gate.py predict ...`)"
    pid = m.group(1)
    preds = led.predictions()
    if pid not in preds:
        return f"derivation gate P1: prediction {pid} does not exist"
    pred = preds[pid]
    if "result" in pred:
        return f"derivation gate P1: prediction {pid} is closed ({pred['result']['outcome']}); a new run needs a new prediction"
    norm = " ".join(cmd.split())
    runs = led.runs()
    mine = [r for r in runs if r["prediction"] == pid]
    if mine and any(r["command"] != norm for r in mine):
        return f"derivation gate P1: prediction {pid} already admitted a different command; one prediction, one command (identical repeats only)"
    if len(mine) >= MAX_REPEATS:
        return f"derivation gate P1: prediction {pid} reached {MAX_REPEATS} repeats"
    sk = skeleton(cmd)
    for r in runs:
        other = preds.get(r["prediction"])
        if r["prediction"] != pid and other and r["skeleton"] == sk and other["basis"] == pred["basis"]:
            return f"derivation gate P2: sweep; {r['prediction']} ran the same command shape on the same basis; a new value needs a new basis"
    led._append(led.runs_path, {"prediction": pid, "command": norm, "skeleton": sk, "t": int(time.time())})
    return None


def cli_only(cmd):
    """True when the command is exactly one invocation of this script's CLI, with no shell operators."""
    if not CLI_RE.match(cmd) or "$(" in cmd or "`" in cmd:
        return False
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    try:
        toks = list(lex)
    except ValueError:
        return False
    return not any(t and set(t) <= set(";&|<>()") for t in toks)


def decide_write(inp, led):
    path = str(inp.get("file_path") or inp.get("notebook_path") or "")
    if any(p in path for p in PROTECTED):
        return f"derivation gate: {path} is a protected ledger/gate file"
    return None


def hook(led):
    try:
        data = json.load(sys.stdin)
        tool, inp = data.get("tool_name", ""), data.get("tool_input", {}) or {}
        if tool == "Bash":
            refusal = decide_bash(inp, led)
        elif tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            refusal = decide_write(inp, led)
        else:
            refusal = None
    except Exception as e:  # fail closed
        refusal = f"derivation gate: internal error, refusing: {type(e).__name__}: {e}"
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    return 0


def predict(led, a):
    ok, why = basis_ok(a["basis"], led.root)
    if not ok:
        raise SystemExit(f"refused: {why}")
    for k in ("q", "question", "predicted", "falsifier"):
        if not a[k].strip():
            raise SystemExit(f"refused: --{k} is empty")
    if int(a["wall_s"]) <= 0:
        raise SystemExit("refused: --wall-s must be positive")
    preds = led.predictions()
    open_ = [p for p in preds.values() if "result" not in p]
    if len(open_) >= MAX_OPEN:
        raise SystemExit(f"refused: {len(open_)} predictions have no result ({', '.join(p['id'] for p in open_)}); record outcomes first")
    pid = f"{a['q']}-{len([p for p in preds.values() if p['q'] == a['q']]) + 1}"
    rec = {"kind": "prediction", "id": pid, "q": a["q"], "question": a["question"], "predicted": a["predicted"],
           "basis": a["basis"], "basis_sha": hashlib.sha256(a["basis"].encode()).hexdigest()[:16],
           "falsifier": a["falsifier"], "wall_s": int(a["wall_s"]), "t": int(time.time())}
    led._append(led.preds_path, rec)
    print(pid)


def result(led, pid, outcome, measured, receipt):
    preds = led.predictions()
    if pid not in preds:
        raise SystemExit(f"refused: no prediction {pid}")
    if "result" in preds[pid]:
        raise SystemExit(f"refused: {pid} already has a result")
    if outcome not in ("confirmed", "refuted", "withdrawn"):
        raise SystemExit("refused: outcome is confirmed|refuted|withdrawn")
    if outcome != "withdrawn" and not any(r["prediction"] == pid for r in led.runs()):
        raise SystemExit(f"refused: {pid} admitted no run; only `withdrawn` is possible")
    led._append(led.preds_path, {"kind": "result", "id": pid, "outcome": outcome, "measured": measured,
                                 "receipt": receipt, "t": int(time.time())})


def status(led):
    runs = led.runs()
    for p in led.predictions().values():
        n = sum(1 for r in runs if r["prediction"] == p["id"])
        res = p.get("result", {})
        print(f"{p['id']}\t{res.get('outcome', 'OPEN')}\truns={n}\twall_est={p['wall_s']}s\tbasis={p['basis'][:60]}\tmeasured={res.get('measured', '')}")


def parse_kv(argv):
    out, i = {}, 0
    while i < len(argv):
        if not argv[i].startswith("--") or i + 1 >= len(argv):
            raise SystemExit(f"bad argument {argv[i]}")
        out[argv[i][2:].replace("-", "_")] = argv[i + 1]
        i += 2
    return out


def self_test():
    """Each control must refuse; each positive must admit. Runs on a throwaway ledger."""
    fails = []
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / ".intent").mkdir()
        (root / ".intent/runs.toml (unpublished)").write_text((ROOT / ".intent/runs.toml (unpublished)").read_text())
        (root / "T.lean").write_text("theorem t : True := trivial\n")
        led = Ledger(root)

        def expect(name, refusal, want_refuse):
            if bool(refusal) != want_refuse:
                fails.append(f"{name}: expected {'refusal' if want_refuse else 'admit'}, got {refusal!r}")
            else:
                print(f"ok {'control' if want_refuse else 'positive'}[{name}]")

        run = "cargo-gate.sh /tmp/x.log run --release --example hotcold_decode -- --budget 2048"
        expect("no-prediction", decide_bash({"command": run, "run_in_background": True}, led), True)
        expect("fg-long", decide_bash({"command": "ls", "timeout": 570_000}, led), True)
        expect("fg-run-default-timeout", decide_bash({"command": "PREDICTION=x " + run}, led), True)
        expect("gate-not-run-class", decide_bash({"command": "cargo test --workspace", "run_in_background": True}, led), False)
        expect("scratch-probe", decide_bash({"command": "python3 /tmp/a/scratchpad/pairing.py m g 0", "run_in_background": True}, led), True)
        expect("ledger-append-bash", decide_bash({"command": "echo {} >> .intent/predictions.jsonl (unpublished)"}, led), True)
        expect("ledger-cli-chained", decide_bash({"command": "python3 .claude/hooks/derivation_gate.py status; echo x > .intent/runs.jsonl (unpublished)"}, led), True)
        expect("ledger-cli", decide_bash({"command": "python3 .claude/hooks/derivation_gate.py status"}, led), False)
        expect("ledger-write", decide_write({"file_path": str(root / ".intent/runs.jsonl (unpublished)")}, led), True)
        expect("hook-edit", decide_write({"file_path": "/x/.claude/hooks/derivation_gate.py"}, led), True)
        expect("settings-edit", decide_write({"file_path": "/x/.claude/settings.json"}, led), True)

        def pred(q, basis):
            buf = sys.stdout
            sys.stdout = out = __import__("io").StringIO()
            try:
                predict(led, {"q": q, "question": "does budget matter", "predicted": "KLD < 0.05",
                              "basis": basis, "falsifier": "KLD >= 0.05", "wall_s": "300"})
            finally:
                sys.stdout = buf
            return out.getvalue().strip()

        for name, basis in (("basis-missing-path", "nope/x.lean:3"), ("basis-past-eof", "T.lean:9"), ("basis-empty-unproved", "unproved:   ")):
            try:
                pred("bad", basis)
                expect(name, None, True)
            except SystemExit as e:
                expect(name, str(e), True)
        p1 = pred("budget", "T.lean:1 -- the cut bound")
        bg = {"run_in_background": True}
        expect("predicted-run", decide_bash({"command": f"PREDICTION={p1} " + run, **bg}, led), False)
        expect("repeat-identical", decide_bash({"command": f"PREDICTION={p1}  " + run, **bg}, led), False)
        expect("same-prediction-new-value", decide_bash({"command": f"PREDICTION={p1} " + run.replace("2048", "4096"), **bg}, led), True)
        p2 = pred("budget", "T.lean:1 -- the cut bound")
        expect("sweep-same-basis", decide_bash({"command": f"PREDICTION={p2} " + run.replace("2048", "4096"), **bg}, led), True)
        p3 = pred("budget", "unproved: the far mass above rank 4096 is below 0.01 of the total, so doubling halves KLD")
        expect("new-basis-new-value", decide_bash({"command": f"PREDICTION={p3} " + run.replace("2048", "4096"), **bg}, led), False)
        try:
            pred("budget", "T.lean:1")
            expect("open-cap", None, True)
        except SystemExit as e:
            expect("open-cap", str(e), True)
        result(led, p1, "refuted", "KLD 0.31", "")
        expect("closed-prediction", decide_bash({"command": f"PREDICTION={p1} " + run, **bg}, led), True)
        try:
            result(led, p2, "confirmed", "x", "")
            expect("result-without-run", None, True)
        except SystemExit as e:
            expect("result-without-run", str(e), True)
        sys.stdin = __import__("io").StringIO("not json")
        expect("fail-closed", hook(led) == 2 and "refuse", True)
    if fails:
        print("\n".join(fails))
        return 1
    print("derivation gate self-test: all controls refuse, all positives admit")
    return 0


def main():
    led = Ledger(ROOT)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "hook"
    if cmd == "hook":
        return hook(led)
    if cmd == "predict":
        predict(led, parse_kv(sys.argv[2:]))
    elif cmd == "result":
        if len(sys.argv) < 4:
            raise SystemExit("usage: result ID OUTCOME --measured TEXT [--receipt PATH]")
        kv = parse_kv(sys.argv[4:])
        result(led, sys.argv[2], sys.argv[3], kv.get("measured", ""), kv.get("receipt", ""))
    elif cmd == "status":
        status(led)
    elif cmd == "self-test":
        return self_test()
    else:
        raise SystemExit(f"unknown command {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
