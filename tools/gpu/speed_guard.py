#!/usr/bin/env python3
"""SPEED-GUARD: paired, load-robust refusal of any tg128/pp512 regression below the SOTA2 receipt.

v2 rule (interleaved paired design). The reference binary is the receipt binary
(`/home/socol/oc-tmp/sota2/apeiron`, sha256 recorded at start); the candidate is `gpu_decode_bench`
built from this tree. Each pair runs the reference and the candidate in alternating order
(ref->cand, cand->ref, ...), both arms running tg128 and pp512, the whole pair holding
`/tmp/opencode/gpu-time.lock` so a shared box load lands on both arms. Per metric the per-pair
ratio is

  r_i = cand_i / ref_i   (i = 1..N, N = --pairs, default 6)

and the metric FAILS when either bound holds:

  mean_r + 2*SE < 1.0   (statistically slower than the reference; SE = sd(r_i)/sqrt(N))
  mean_r      < 0.97    (material regression regardless of noise)

FAIL if any candidate SEQHASH != 9092dbbaff119a2e or != the reference's SEQHASH in the same pair.
Exit 0 when both metrics pass.

v1's absolute table (receipt mean, measured cand mean, measured ref mean) is retained as an
**informational** line; it never decides the verdict. The whole-run wall-clock ceiling stays at
BUDGET_S and exhaustion prints `FAIL BUDGET EXPIRED`.

The paired ratio is the load answer: v1 compared a fresh mean against `receipt_mean - receipt_sd`
and reddened on unchanged code under box load (tg128 is dispatch-latency-bound, so absolute tok/s
shifts with load). The ratio cancels the shared load; the two bounds catch a shift that moves the
candidate relative to the reference.

Receipt constants derived from the SOTA2 apeiron logs (not the receipt prose), 6 rounds,
sample sd (n-1). Logs and their sha256:

  /home/socol/oc-tmp/sota2/r1-ap-tg.log 6b5ebbf875a9da71b530c4a7840b67a4785700c43d4b861bbb5e6734dce63aca
  /home/socol/oc-tmp/sota2/r2-ap-tg.log c91c33b6546688c8f6b3faf221ea0d99773e11ae7806e022a0ae5a0ca937343e
  /home/socol/oc-tmp/sota2/r3-ap-tg.log 072ed9c93aa0aab0cd98ff2ecc35717a31eea255d17d0c6390a84d1b0c3d785f
  /home/socol/oc-tmp/sota2/r4-ap-tg.log ea843759b470da71865485247903176e134eb1fdd098349c08afc0646e1c7eab
  /home/socol/oc-tmp/sota2/r5-ap-tg.log e41ed3b070049e88eddd639f968d7b710efb14710eb7a94e7da69863e522cbeb
  /home/socol/oc-tmp/sota2/r6-ap-tg.log 84c0ac920a512319f76f2100fbcb9996801c95b951fc1676d2cb337577a49d86
  /home/socol/oc-tmp/sota2/r1-ap-pp.log 4604a6c31966d6e4e9221096a37658beee1a82d63bb4f45aa65c9de40589e883
  /home/socol/oc-tmp/sota2/r2-ap-pp.log 175bd5e64d116b732f2ec557d8451a29f48813b435014f92a50127727c5985bb
  /home/socol/oc-tmp/sota2/r3-ap-pp.log a1aaddb50fa68306d3a35b3ab3feff5bd3f735239f68a0d995484ec1bb8a093b
  /home/socol/oc-tmp/sota2/r4-ap-pp.log b128f41d1563a257b95dc5c8a6aef28b07b4ee8962191432e6942a79842269c9
  /home/socol/oc-tmp/sota2/r5-ap-pp.log 976674988d9dc2f3f3c8620647a3bcf74f5e8397e41a1683588e5ca0cc9ba0d3
  /home/socol/oc-tmp/sota2/r6-ap-pp.log 3a86316f85a62a12ddeddffd18e8e978e7572360e76d3cae73cd755ed26689d7

Per-round means parsed from the `BENCH carrier=q4 tg128/pp512 ...` lines:
  tg: [901.8, 926.9, 886.7, 871.3, 898.1, 879.7]   mean 894.083333  sample sd 19.662596
  pp: [108771.2, 108853.4, 108356.4, 110600.9, 111051.1, 110399.1]
                                                    mean 109672.016667 sample sd 1140.683689
SEQHASH = 9092dbbaff119a2e over every apeiron run.

Protocol: SOTA2-RECEIPT.md, /home/socol/oc-tmp/sota2/run.sh.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

# --- receipt constants (derived above; do not re-derive) -------------------------------------
RECEIPT_TG_MEAN = 894.083333
RECEIPT_TG_SD = 19.662596
RECEIPT_PP_MEAN = 109672.016667
RECEIPT_PP_SD = 1140.683689
SEQHASH_EXPECTED = "9092dbbaff119a2e"

MODEL = "/home/socol/models/gguf/qwen05b-Q4_K_M.gguf"
REFERENCE_BIN = "/home/socol/oc-tmp/sota2/apeiron"
CARGO_GATE = "/home/socol/oc-tmp/bin/cargo-gate.sh"
GPU_LOCK = "/tmp/opencode/gpu-time.lock"
MEM_CAP = "6G"

BUDGET_S = 570.0          # whole-run wall-clock ceiling (§10)
PER_RUN_TIMEOUT = 200     # per-run `timeout 200` inside the systemd scope

# --- v2 decision rule -------------------------------------------------------------------------
MATERIAL_RATIO = 0.97     # mean_r below this is a material regression regardless of noise
SIGMA = 2.0               # number of standard errors for the statistical bound

# tg128/pp512 arms, exactly as /home/socol/oc-tmp/sota2/run.sh invokes the receipt binary.
TG_ARGS = ["--gen", "128", "--reps", "5", "--carrier", "q4"]
PP_ARGS = ["--prefill", "512", "--reps", "5", "--carrier", "q4"]
PP_ENV: dict[str, str] = {}

TG_RE = re.compile(r"tg(\d+): mean ([\d.]+) tok/s min ([\d.]+) max ([\d.]+) over (\d+) reps")
PP_RE = re.compile(r"pp(\d+): mean ([\d.]+) tok/s min ([\d.]+) max ([\d.]+) over (\d+) reps")
SEQ_RE = re.compile(r"SEQHASH\s+([0-9a-fA-F]+)\s+over")


class Expired(Exception):
    """Wall-clock budget exhausted."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> float:
    return time.monotonic()


def check_deadline(deadline: float, stage: str) -> None:
    if now() >= deadline:
        raise Expired(f"BUDGET EXPIRED: {BUDGET_S:.0f}s ceiling exceeded before {stage}")


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_binary(log_path: str) -> str:
    """touch src/lib.rs so the binary binds to this tree's source, then cargo-gate build."""
    root = repo_root()
    lib = os.path.join(root, "src", "lib.rs")
    os.utime(lib, None)
    env = dict(os.environ)
    env["PATH"] = "/home/socol/oc-tmp/bin:" + env.get("PATH", "")
    env.pop("CARGO_TARGET_DIR", None)
    cmd = [CARGO_GATE, log_path, "build", "--release", "--locked",
           "--features", "gpu", "--example", "gpu_decode_bench"]
    with open(log_path, "wb") as f:
        proc = subprocess.run(cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"build failed (rc={proc.returncode}); see {log_path}")
    built = "/home/socol/oc-tmp/shared-target/release/examples/gpu_decode_bench"
    if not os.path.exists(built):
        raise SystemExit(f"build produced no binary at {built}")
    return built


def run_arm(bin_path: str, args: list[str], env_extra: dict[str, str],
            deadline: float, log_path: str) -> str:
    """Run one arm under the systemd scope + per-run timeout. The caller holds GPU_LOCK."""
    check_deadline(deadline, f"run {os.path.basename(bin_path)} {args[:2]}")
    env = dict(os.environ)
    env.update(env_extra)
    inner = ["systemd-run", "--user", "--scope", "-q",
             "-p", f"MemoryMax={MEM_CAP}", "-p", "MemorySwapMax=0",
             "timeout", str(PER_RUN_TIMEOUT), bin_path, MODEL] + args
    proc = subprocess.run(inner, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=PER_RUN_TIMEOUT + 30)
    out = proc.stdout.decode("utf-8", "replace")
    with open(log_path, "w") as f:
        f.write(out)
    return out


def parse_seqhash(out: str) -> str | None:
    m = SEQ_RE.search(out)
    return m.group(1) if m else None


def parse_bench(out: str, regex: re.Pattern) -> float | None:
    m = regex.search(out)
    return float(m.group(2)) if m else None


def fmt(x: float) -> str:
    return f"{x:,.1f}"


def ratio_stats(rs: list[float]) -> tuple[float, float, float, float, float]:
    m = statistics.mean(rs)
    sd = statistics.stdev(rs) if len(rs) > 1 else 0.0
    se = sd / math.sqrt(len(rs)) if len(rs) > 1 else 0.0
    return m, sd, se, min(rs), max(rs)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="paired guard against the SOTA2 receipt binary (v2)")
    ap.add_argument("--pairs", type=int, default=6, help="interleaved pairs (default 6)")
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--bin", dest="bin_path", default=None,
                    help="candidate binary (skip the build)")
    args = ap.parse_args()

    if args.pairs < 2:
        raise SystemExit(f"--pairs must be >= 2 for a sample sd (got {args.pairs})")

    t_start = now()
    deadline = t_start + BUDGET_S
    tmpdir = tempfile.mkdtemp(prefix="speed-guard-")
    logdir = os.path.join(tmpdir, "logs")
    os.makedirs(logdir, exist_ok=True)

    ref_sha = sha256_file(REFERENCE_BIN)
    print(f"[speed-guard] v2 paired guard: model {MODEL}")
    print(f"[speed-guard] budget {BUDGET_S:.0f}s, pairs {args.pairs}")
    print(f"[speed-guard] reference {REFERENCE_BIN} sha256 {ref_sha}")
    print(f"[speed-guard] rule: FAIL if mean_r + {SIGMA:.1f}*SE < 1.000 "
          f"(statistically slower) or mean_r < {MATERIAL_RATIO:.3f} (material)")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(),
                          capture_output=True, text=True).stdout.strip()

    if args.bin_path:
        cand_bin = os.path.abspath(args.bin_path)
        cand_sha = sha256_file(cand_bin)
    else:
        try:
            built = build_binary(os.path.join(tmpdir, "build.log"))
        except Expired as e:
            print(f"FAIL {e}")
            return 1
        cand_bin = os.path.join(tmpdir, "gpu_decode_bench")
        shutil.copy2(built, cand_bin)
        cand_sha = sha256_file(cand_bin)
    print(f"[speed-guard] HEAD {head}")
    print(f"[speed-guard] candidate {cand_bin} sha256 {cand_sha}")

    pairs: list[dict] = []
    fails: list[str] = []
    errors: list[str] = []

    lock_fd = os.open(GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        for i in range(1, args.pairs + 1):
            order = ["ref", "cand"] if i % 2 == 1 else ["cand", "ref"]
            check_deadline(deadline, f"pair {i}")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)      # whole pair, not per run
            arm: dict[str, dict] = {}
            try:
                for which in order:
                    binp = REFERENCE_BIN if which == "ref" else cand_bin
                    try:
                        out_tg = run_arm(binp, TG_ARGS, {},
                                         deadline, os.path.join(logdir, f"r{i}-{which}-tg.log"))
                        out_pp = run_arm(binp, PP_ARGS, PP_ENV,
                                         deadline, os.path.join(logdir, f"r{i}-{which}-pp.log"))
                    except Expired:
                        raise
                    except subprocess.TimeoutExpired:
                        errors.append(f"pair {i} {which}: run exceeded {PER_RUN_TIMEOUT}s")
                        arm[which] = {"tg": None, "pp": None, "seq": None}
                        continue
                    arm[which] = {
                        "tg": parse_bench(out_tg, TG_RE),
                        "pp": parse_bench(out_pp, PP_RE),
                        "seq": parse_seqhash(out_tg),
                    }
            except Expired:
                print(f"FAIL BUDGET EXPIRED: {BUDGET_S:.0f}s ceiling reached in pair {i}")
                return 1
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

            ref = arm.get("ref", {"tg": None, "pp": None, "seq": None})
            cand = arm.get("cand", {"tg": None, "pp": None, "seq": None})
            for which, a in (("ref", ref), ("cand", cand)):
                if a["tg"] is None:
                    errors.append(f"pair {i} {which}: no tg128 BENCH line")
                if a["pp"] is None:
                    errors.append(f"pair {i} {which}: no pp512 BENCH line")

            # SEQHASH: candidate must equal the receipt value AND the reference's in this pair.
            for which, a in (("ref", ref), ("cand", cand)):
                s = a["seq"]
                if s is None:
                    errors.append(f"pair {i} {which}: no SEQHASH line")
                elif s != SEQHASH_EXPECTED:
                    fails.append(f"FAIL SEQHASH: pair {i} {which} {s} != {SEQHASH_EXPECTED}")
            if ref["seq"] is not None and cand["seq"] is not None and ref["seq"] != cand["seq"]:
                fails.append(f"FAIL SEQHASH: pair {i} candidate {cand['seq']} != reference {ref['seq']}")

            pairs.append({"ref": ref, "cand": cand})

            tg_r = (cand["tg"] / ref["tg"]) if (ref["tg"] and cand["tg"]) else None
            pp_r = (cand["pp"] / ref["pp"]) if (ref["pp"] and cand["pp"]) else None
            print(f"[speed-guard] pair {i} order {'->'.join(order)}: "
                  f"tg ref {fmt(ref['tg']) if ref['tg'] else 'NA'} cand "
                  f"{fmt(cand['tg']) if cand['tg'] else 'NA'} r "
                  f"{f'{tg_r:.4f}' if tg_r is not None else 'NA'} | "
                  f"pp ref {fmt(ref['pp']) if ref['pp'] else 'NA'} cand "
                  f"{fmt(cand['pp']) if cand['pp'] else 'NA'} r "
                  f"{f'{pp_r:.4f}' if pp_r is not None else 'NA'}")
    finally:
        os.close(lock_fd)

    check_deadline(deadline, "final report")

    # --- per-metric paired ratios -------------------------------------------------------------
    def ratios(metric: str) -> list[float]:
        out = []
        for p in pairs:
            r, c = p["ref"][metric], p["cand"][metric]
            if r and c:
                out.append(c / r)
        return out

    tg_rs, pp_rs = ratios("tg"), ratios("pp")
    tg_m, tg_sd, tg_se, tg_min, tg_max = ratio_stats(tg_rs)
    pp_m, pp_sd, pp_se, pp_min, pp_max = ratio_stats(pp_rs)
    tg_bound, pp_bound = tg_m + SIGMA * tg_se, pp_m + SIGMA * pp_se

    for name, m, bound in (("tg128", tg_m, tg_bound), ("pp512", pp_m, pp_bound)):
        if bound < 1.0:
            fails.append(f"FAIL {name}: statistically slower — mean_r {m:.4f} + "
                         f"{SIGMA:.1f}*SE {SIGMA * (tg_se if name == 'tg128' else pp_se):.4f}"
                         f" = {bound:.4f} < 1.000")
        if m < MATERIAL_RATIO:
            fails.append(f"FAIL {name}: material regression — mean_r {m:.4f} < {MATERIAL_RATIO:.3f}")
    fails.extend(errors)

    # --- informational absolute table (v1's numbers; never decides the verdict) ---------------
    ref_tg_mean = statistics.mean([p["ref"]["tg"] for p in pairs if p["ref"]["tg"]]) if any(
        p["ref"]["tg"] for p in pairs) else 0.0
    ref_pp_mean = statistics.mean([p["ref"]["pp"] for p in pairs if p["ref"]["pp"]]) if any(
        p["ref"]["pp"] for p in pairs) else 0.0
    cand_tg_mean = statistics.mean([p["cand"]["tg"] for p in pairs if p["cand"]["tg"]]) if any(
        p["cand"]["tg"] for p in pairs) else 0.0
    cand_pp_mean = statistics.mean([p["cand"]["pp"] for p in pairs if p["cand"]["pp"]]) if any(
        p["cand"]["pp"] for p in pairs) else 0.0

    print()
    print(f"[speed-guard] rule: FAIL if (mean_r + {SIGMA:.1f}*SE < 1.000) "
          f"or (mean_r < {MATERIAL_RATIO:.3f}); bounds below")
    hdr = (f"{'metric':<7} {'mean_r':>8} {'sd':>8} {'SE':>8} {'2*SE':>8} "
           f"{'min':>7} {'max':>7} {'mean+2SE':>9} {'<1.000?':>8} {'<0.970?':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for name, m, sd, se, lo, hi, bound in (
        ("tg128", tg_m, tg_sd, tg_se, tg_min, tg_max, tg_bound),
        ("pp512", pp_m, pp_sd, pp_se, pp_min, pp_max, pp_bound),
    ):
        v = "FAIL" if (bound < 1.0 or m < MATERIAL_RATIO) else "PASS"
        print(f"{name:<7} {m:>8.4f} {sd:>8.4f} {se:>8.4f} {SIGMA * se:>8.4f} "
              f"{lo:>7.4f} {hi:>7.4f} {bound:>9.4f} "
              f"{'YES' if bound < 1.0 else 'no':>8} "
              f"{'YES' if m < MATERIAL_RATIO else 'no':>8}  {v}")
    print()
    print("[speed-guard] informational (absolute tok/s; never decides the verdict)")
    ihdr = (f"{'metric':<7} {'receipt_mean':>13} {'cand_mean':>11} {'ref_mean':>11}")
    print(ihdr)
    print("-" * len(ihdr))
    print(f"{'tg128':<7} {RECEIPT_TG_MEAN:>13,.1f} {cand_tg_mean:>11,.1f} {ref_tg_mean:>11,.1f}")
    print(f"{'pp512':<7} {RECEIPT_PP_MEAN:>13,.1f} {cand_pp_mean:>11,.1f} {ref_pp_mean:>11,.1f}")
    print()

    elapsed = now() - t_start
    print(f"[speed-guard] seqhash expected {SEQHASH_EXPECTED}")
    for line in fails:
        print(line)
    print(f"[speed-guard] elapsed {elapsed:.1f}s / {BUDGET_S:.0f}s")

    if args.json_path:
        payload = {
            "version": "v2",
            "head": head,
            "reference": {"path": REFERENCE_BIN, "sha256": ref_sha},
            "candidate": {"path": cand_bin, "sha256": cand_sha},
            "model": MODEL,
            "seqhash_expected": SEQHASH_EXPECTED,
            "pairs": args.pairs,
            "rule": {"sigma": SIGMA, "material_ratio": MATERIAL_RATIO},
            "receipt": {"tg128": {"mean": RECEIPT_TG_MEAN, "sd": RECEIPT_TG_SD},
                         "pp512": {"mean": RECEIPT_PP_MEAN, "sd": RECEIPT_PP_SD}},
            "paired": {
                "tg128": {"ratios": tg_rs, "mean_r": tg_m, "sd": tg_sd, "se": tg_se,
                          "min": tg_min, "max": tg_max, "stat_bound": tg_bound},
                "pp512": {"ratios": pp_rs, "mean_r": pp_m, "sd": pp_sd, "se": pp_se,
                          "min": pp_min, "max": pp_max, "stat_bound": pp_bound},
            },
            "informational": {
                "tg128": {"receipt_mean": RECEIPT_TG_MEAN, "cand_mean": cand_tg_mean,
                          "ref_mean": ref_tg_mean},
                "pp512": {"receipt_mean": RECEIPT_PP_MEAN, "cand_mean": cand_pp_mean,
                          "ref_mean": ref_pp_mean},
            },
            "verdict": "FAIL" if fails else "PASS",
            "failures": fails,
            "elapsed_s": elapsed,
        }
        with open(args.json_path, "w") as f:
            json.dump(payload, f, indent=2)

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
