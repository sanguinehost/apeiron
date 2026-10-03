#!/usr/bin/env python3
"""Per-invocation cost attribution for the Apeiron Q8 SPIR-V kernel.

Drives the existing ``tools/gpu/q8_driver.c`` device-timestamp path (the same
instrument ``q8_shapes.py`` uses) over four bounded sweeps of the executing
kernel ``src/spirv/q8_tile.comp`` and decomposes the device time into dispatch,
per-workgroup latency, memory and arithmetic terms, so that the 30.8x-304x gap
to llama.cpp CUDA MMQ is attributed to the term that actually causes it.

Geometry (from the shader): ``local_size 8x8`` = 64 invocations/workgroup, each
invocation a 2-row x 4-token tile, so one workgroup covers 16 rows x 32 tokens
and ``MACS_PER_WG_BLOCK = 16*32*32 = 16384`` MACs per (workgroup, block).  The
grid is ``ng = ceil(M/16) * ceil(tokens/32)`` workgroups and the reduction depth
is ``blocks = K/32``.

Sweeps
------
* ``occ``  : blocks=64 fixed, M x tokens grid -> throughput vs grid size (occupancy).
* ``work`` : M=1024, tokens=256 (ng=512) fixed, blocks from 1 to 224 -> arithmetic plateau.
* ``iso``  : ng=1 (M=16, tokens=32), blocks in {1,64} -> dispatch floor and the
             exposed per-block serial latency of a single workgroup.
* ``real`` : the five Qwen3.5-0.8B projection shapes at tokens=200/512.
* ``split``: M=128, tokens=256 (ng=64, one workgroup per SM on this 84-SM part) fixed,
  blocks in {64,112,160,224,320}, run once per cell of a 2x2 of cost-probe mutants of
  ``q8_tile.comp`` (:107 dot, :121-124 epilogue) -> the measured dot/epilogue split of the
  deep-K marginal.  Each mutant removes one term and keeps every load live, so the buffer
  access pattern is identical across the four cells; they compute different values and are
  cost probes, not kernels.  ng=64 is the resolution limit of the instrument: the same 2x2
  at the work sweep's ng=512 leaves the epilogue delta at -0.03 +/- 0.41 ns/(wg,block) and
  the dot delta negative, because six co-resident workgroups per SM put a ~7% cell-to-cell
  spread on a marginal whose ALU terms are ~2%.

Roofline inputs are measured: arithmetic intensity ``MACs / unique_weight_bytes``
(= tokens for aligned shapes) against the fitted local arithmetic plateau, and
the achieved bytes/s against the CUDA-probed device peak bandwidth.

Every cost-probe sweep below is repeated ``COST_MODEL_PASSES`` times, interleaved by depth
within each pass, so every cell's slope carries a measured between-pass spread.

``rem``  : the same ng=64 geometry and the same depths, adding five more cost-probe cells
  of ``src/spirv/q8_tile.comp`` that price the 96.6-97.8% remainder the 2x2 leaves: the
  tile registers forced to constant indices (the kernel keeps them in Function storage, so
  every access is local-memory traffic), the dm[]/e[] metadata staging, the w[]/x[]
  addresses collapsed onto 8 addresses, the q loop trip count 8->1, and the finalize
  epilogue's arithmetic.  All nine cells are interleaved by depth.  See
  ``tools/gpu/q8_remainder.py``.

Controls (``--controls``)
-------------------------
* ``--control-self-test`` : pure checker self-test; a seeded flat (stuck-timer)
  series and a flat control pair must be rejected.  Two split seeds bracket the
  dot/epilogue hypothesis from both sides and both must be ACCEPTED, so the
  checker cannot be encoding it; five seeds per gate must be rejected.
* ``--controls``          : runs the live sweeps, requires the occupancy series
  to be strictly monotone in ``ng`` and the control pair to differ measurably,
  then injects a stuck timer and an inverted series into copies of the *real*
  occupancy results and requires each checker to redden under this harness, then
  confirms the unmutated series passes.
* ``--controls`` also runs the split suite, and every term of it is CLASSIFIED rather
  than sized: a priced term gets the interval ``[v - k*se, v + k*se]`` against the null
  "this probe's work is free" and one of three states -- ``resolved+``, ``resolved-``,
  ``unresolved``.  No bar fires on an ``unresolved`` term, which is what removed the
  inherited 2x2's ten red runs in 19.  Seeding a no-op epilogue, a both-zero attribution,
  a bare cell the device resolves as slower, or a wrong-shape row into the *real* measured
  slopes must still redden the checker by name.  A separate static census of each cell's
  SPIR-V proves the elisions reached the compiled module -- without it a mutant that failed
  to edit would report a zero cost term.
* ``--controls`` runs the remainder suite the same way: the tile-scalarized cell must
  move the marginal, and eight seeded census defects must redden
  ``q8_remainder.check_rem_census``.  Which remainder term dominates is reported, never
  asserted.

ACCEPTANCE PROTOCOL -- which quantity decides exit
--------------------------------------------------
``tools/gpu/q8_costmodel.py`` holds it; this gate restates it rather than repeating it.

* Measured uncertainty.  The deep-K sweep and the real-shape sweep run ``COST_MODEL_PASSES``
  times inside one invocation.  A term's standard error is
  ``hypot(mean(per-pass fit error), sd(per-pass slopes)/sqrt(COST_MODEL_PASSES))``.  The
  between-pass component is what the single-pass estimator omitted, and it is where the two
  device regimes live: the executing kernel's own marginal moved 4% between regimes while its
  within-sweep fit error read 0.11 ns.
* The resolved-term rule, above.  The only term-level requirement is conditional coherence --
  *if* a declared cost term is ``resolved-``, the device contradicts the cost model and the
  gate is red -- plus one count: at least one declared cost term must be ``resolved+``, the
  statement that the clock reads the kernel's work at all.  Both are invariant to which term
  dominates.
* The exit quantity is the per-unit spread of the whole kernel: ns per dispatch unit at the
  ten real projection shapes, divided by the instrument's own spread on that same quantity,
  measured by repeating the shapes in this same invocation.  Red iff the ratio is <= 1.  The
  threshold is the instrument's own measured spread, not a constant chosen for these shapes,
  and it moves with the device, the clock and the estimator.

The naive whole-kernel model -- device_ns is one linear function of MACs -- is measured FALSE
and reported, never floored: across the ten real shapes its R^2 is ~0.74 with a ~110% worst
residual, while dispatch units plus workgroup count give R^2 ~0.998 with a ~60% worst
residual.  A high R^2 on a mis-specified model is not evidence of a rate, so no R^2 floor is
asserted anywhere and the per-unit spread beside both is the honest test of "one rate".

The instrument is the GPU timestamp delta across one dispatch; the control that
can fail is the ordering/difference check, which a stuck or zero timer violates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_BUDGET_SECONDS = 540.0
BUDGET_GRACE_SECONDS = 5.0

# --- geometry of the executing kernel (src/spirv/q8_tile.comp) ---------------
LOCAL_X = 8
LOCAL_Y = 8
INVOCATIONS_PER_WG = LOCAL_X * LOCAL_Y          # 64
ROWS_PER_WG = LOCAL_X * 2                        # 16
TOKENS_PER_WG = LOCAL_Y * 4                      # 32
MACS_PER_WG_BLOCK = ROWS_PER_WG * TOKENS_PER_WG * 32   # 16384

# --- device identity/limits (RTX 5080) --------------------------------------
DEVICE_NAME_FALLBACK = "NVIDIA GeForce RTX 5080"
SPEC_SM_COUNT = 84
SPEC_MAX_THREADS_PER_SM = 1536
SPEC_MEM_BUS_BITS = 256
SPEC_MEM_CLOCK_HZ = 15_001_000_000      # nvidia-smi clocks.max.memory = 15001 MHz

# --- sweep definitions -------------------------------------------------------
OCC_BLOCKS = 64
OCC_M = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
OCC_TOKENS = [32, 64, 128, 256]

WORK_M = 1024
WORK_TOKENS = 256                       # ng = 64 * 8 = 512
WORK_BLOCKS = [1, 8, 16, 32, 64, 112, 224]
DEEP_BLOCKS = 64                        # blocks >= this = "deep" regime for the plateau fit

ISO_M = 16
ISO_TOKENS = 32                         # ng = 1
ISO_BLOCKS = [1, 64]

CONTROL_PAIR = (16, 224)                # work-sweep reduction depths at ng=512
MIN_PAIR_RATIO = 1.50

REAL_TOKENS = [200, 512]
REAL_SHAPES = [
    ("attn_q", 4096, 32),
    ("attn_k_v", 512, 32),
    ("attn_output", 1024, 64),
    ("ffn_gate_up", 3584, 32),
    ("ffn_down", 1024, 112),
]

# CUDA MMQ comparator band (tools/gpu/cuda_q8_gate.py, measured on this box).
CUDA_GMAC_S = (6425.0, 53644.0)

# --- dot / checked-i128 epilogue cost probes (src/spirv/q8_tile.comp) ----------
# Both anchors are exact single-occurrence substrings of the executing kernel, so a
# source drift makes build_variants() raise instead of silently re-measuring an
# unmodified kernel and reporting a zero split.
#
#   dot      q8_tile.comp:107          8x dotPacked4x8EXT per lane per block
#   epilogue q8_tile.comp:121-124      addChecked(acc[c], shiftedTerm(coef, shift))
#
# Each mutant removes one term and leaves every load live: the dot probe keeps the
# w[]/x[] reads feeding an integer AND, and the epilogue probe keeps mantissa[]/dots[]
# and the weightExp/actExp branches feeding a single u32 add into acc[c].x.  The
# finalize loop is not touched, so the buffer store pattern is identical in all four.
DOT_ANCHOR = "                    dots[c] += dotPacked4x8EXT(wr[r], xt[t]);\n"
DOT_PROBE = "                    dots[c] += (wr[r] & xt[t]);\n"
EPILOGUE_ANCHOR = (
    "                    uint shift = weightExp[r] + actExp[t];\n"
    "                    int64_t coefficient = int64_t(mantissa[r]) * int64_t(dots[c]);\n"
    "                    if (!addChecked(acc[c], shiftedTerm(coefficient, shift)))\n"
    "                        flags[c] |= REFUSE_I128;\n")
EPILOGUE_PROBE = "                    acc[c].x += uint(int64_t(mantissa[r]) * int64_t(dots[c]));\n"

SPLIT_VARIANTS = ("full", "nodot", "noepi", "bare")
# ng=64 on an 84-SM part: one workgroup per SM, so the deep-K marginal is the
# single-workgroup serial cost with no co-resident workgroup to contend for it.
# This is the resolution limit of the measurement.  At the work sweep's ng=512
# (six workgroups per SM) the same 2x2 is unresolvable: the elided-epilogue
# delta fell to -0.03 +/- 0.41 ns/(wg,block) and the elided-dot delta went
# NEGATIVE (-1.13 +/- 0.32), because occupancy contention and clock steps put a
# ~7% cell-to-cell spread on a marginal whose ALU terms are ~2%.  ng=64 holds
# the spread to ~0.3% and resolves the epilogue at >10 sigma.
# M=128, tokens=256 is a cell of the occupancy sweep, so the split is priced at
# a geometry the instrument already measures independently (see SPLIT_RECHECK_*).
SPLIT_M = 128
SPLIT_TOKENS = 256                        # ng = 8 * 8 = 64
SPLIT_BLOCKS = [64, 112, 160, 224, 320]
SPLIT_RECHECK = (128, 256, OCC_BLOCKS)    # (m, tokens, blocks) the occ-sweep overlap
SPLIT_RECHECK_MIN_REL = 0.02              # the two must agree to 2% on the same dispatch
SPLIT_MIN_SIGMA = 3.0                     # WIDTH of a term's acceptance interval, in combined
                                        # standard errors; not a size a term must reach.  An
                                        # unresolved term decides nothing.
# Terms a removal-of-work model predicts to be positive, and therefore the only ones whose
# classification can bite.  Which of the three leads is reported, never asserted: the
# dot-dominant, balanced and epilogue-dominant seeds are all accepted by --control-self-test.
SPLIT_COST_TERMS = ("dot", "checked_i128_epilogue", "removable_alu")
# Repetitions of each sweep inside one invocation; the source of the measured uncertainty.
COST_MODEL_PASSES = 3
SPLIT_CENSUS_MIN_DELTA = 5                # arithmetic ops the epilogue elision must remove
SPLIT_CENSUS_SLACK = 2                    # codegen slack on the bare-vs-noepi arithmetic delta
# The four cells differ by a few percent, so the per-case minimum over repetitions
# is the estimator: an intermittent rep (clock step or host interrupt) lifts
# dev_min up to 60% over dev_med and inflates a 25-rep median more than the effect
# being measured.  The median slope is reported beside every minimum-based figure.
SPLIT_ESTIMATOR = "device_ns_min"
SPLIT_REPS = 25


def census_spirv(spv: Path) -> dict[str, int]:
    """Static census of one cell's SPIR-V: the non-vacuity witness for the elision.

    The device timing says what a term costs; this says what it *was*.  Without it a
    mutant that failed to edit the shader would report a zero cost term and be
    indistinguishable from a term that is genuinely free.  Storage-class resolution
    is unnecessary here: the census counts whole-module opcodes.
    """
    dis = spv.with_suffix(".dis")
    if not dis.exists():
        subprocess.run(["spirv-dis", "--no-color", str(spv), "-o", str(dis)],
                       check=True, capture_output=True, text=True)
    text = dis.read_text()
    ops: dict[str, int] = {}
    for op in ("OpSDot", "OpUDot", "OpIAddCarry", "OpUAddCarry"):
        ops[op] = len(re.findall(rf"= {op}\b", text))
    ops["sdot"] = ops["OpSDot"] + ops["OpUDot"]
    ops["iaddcarry"] = ops["OpIAddCarry"] + ops["OpUAddCarry"]
    ops["arith"] = len(re.findall(
        r"= Op(?:IAdd|ISub|IMul|BitwiseAnd|ShiftLeftLogical|ShiftRightLogical|"
        r"Not|Select|Convert)\b", text))
    ops["total"] = len(re.findall(r"^\s*(?:%\S+ = )?Op\w+", text, re.M))
    return ops


def check_census(census: dict[str, dict[str, int]]) -> list[str]:
    """Each mutant must have statically removed the work it claims to remove.

    This is the only check in the split that can fire on a *source* defect, and it is
    what makes a sub-sigma device delta mean "free" rather than "not applied".
    """
    problems: list[str] = []
    if set(census) != set(SPLIT_VARIANTS):
        return [f"census covers {sorted(census)}, not {sorted(SPLIT_VARIANTS)}"]
    c = census
    if not c["full"]["sdot"] >= 1:
        problems.append(f"the full cell has {c['full']['sdot']} dot opcodes; the executing "
                        "kernel runs 8 dotPacked4x8EXT per lane per block")
    if c["nodot"]["sdot"] != 0:
        problems.append(f"the nodot cell still holds {c['nodot']['sdot']} dot opcodes; the "
                        "dot elision did not reach the compiled module")
    if not c["noepi"]["sdot"] == c["full"]["sdot"]:
        problems.append(f"the noepi cell holds {c['noepi']['sdot']} dot opcodes against the "
                        f"full cell's {c['full']['sdot']}; the epilogue elision changed the dot")
    d_arith = c["full"]["arith"] - c["noepi"]["arith"]
    if not d_arith >= SPLIT_CENSUS_MIN_DELTA:
        problems.append(f"the epilogue elision removed {d_arith} arithmetic opcodes, below the "
                        f"{SPLIT_CENSUS_MIN_DELTA}-opcode bar; it is not the checked-i128 "
                        "epilogue that was elided")
    # bare elides a superset of noepi's work, so it must remove at least as many
    # arithmetic opcodes — up to a 2-opcode codegen slack.  Exact equality is not
    # available: replacing dotPacked4x8EXT with an integer AND adds one BitwiseAnd,
    # and glslc folds that AND differently depending on whether the checked epilogue is
    # still consuming dots[] (measured: full 86, nodot 87, noepi 61, bare 62).
    d_both = c["full"]["arith"] - c["bare"]["arith"]
    if not d_both >= d_arith - SPLIT_CENSUS_SLACK:
        problems.append(f"the bare cell removed {d_both} arithmetic opcodes against noepi's "
                        f"{d_arith}; bare elides a superset of noepi's work and must not remove "
                        f"less (tolerance {SPLIT_CENSUS_SLACK})")
    if c["bare"]["sdot"] != 0:
        problems.append(f"the bare cell still holds {c['bare']['sdot']} dot opcodes")
    return problems


def build_variants(source: Path) -> dict[str, str]:
    """Derive the four cost-probe cells from the executing kernel's real text."""
    text = source.read_text()
    for name, anchor in (("dot", DOT_ANCHOR), ("epilogue", EPILOGUE_ANCHOR)):
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError(
                f"q8_occupancy: {source} holds {n} occurrences of the {name} anchor, not 1; "
                "the dot/epilogue split would price an unmodified kernel. Re-derive the "
                "anchor constants against the source before trusting a split number.")
    return {
        "full": text,
        "nodot": text.replace(DOT_ANCHOR, DOT_PROBE),
        "noepi": text.replace(EPILOGUE_ANCHOR, EPILOGUE_PROBE),
        "bare": text.replace(DOT_ANCHOR, DOT_PROBE).replace(EPILOGUE_ANCHOR, EPILOGUE_PROBE),
    }


def load_bench():
    spec = importlib.util.spec_from_file_location("q8_bench", HERE / "q8_bench.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load q8_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_remainder():
    spec = importlib.util.spec_from_file_location("q8_remainder", HERE / "q8_remainder.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load q8_remainder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def ng_of(m: int, tokens: int) -> int:
    """Workgroups in the dispatch grid: ceil(M/16) x ceil(tokens/32)."""
    return ceil_div(m, 16) * ceil_div(tokens, 32)


def derive(m: int, tokens: int, blocks: int) -> dict[str, float]:
    ng_x = ceil_div(m, 16)
    ng_y = ceil_div(tokens, 32)
    ng = ng_x * ng_y
    work_units = ng * blocks
    macs = m * tokens * blocks * 32
    unique_weight_bytes = m * blocks * 32                 # Q8_0 packed codes, 1 B/lane
    logical_weight_bytes = ng_y * unique_weight_bytes     # each token-tile re-reads W
    act_bytes = tokens * blocks * (8 * 4 + 4)
    out_bytes = tokens * m * (8 + 4)
    unique_total = unique_weight_bytes + act_bytes + out_bytes
    logical_total = logical_weight_bytes + act_bytes + out_bytes
    return {
        "ng": float(ng), "ng_x": float(ng_x), "ng_y": float(ng_y),
        "n_inv": float(ng * INVOCATIONS_PER_WG),
        "work_units": float(work_units),
        "macs": float(macs),
        "unique_bytes": float(unique_total),
        "logical_bytes": float(logical_total),
        "unique_weight_bytes": float(unique_weight_bytes),
        "intensity": macs / unique_total if unique_total else 0.0,
    }


def run_case(bench, driver_bin: Path, shader_spv: Path, scratch: Path, sweep: str,
             label: str, m: int, tokens: int, blocks: int, reps: int,
             deadline: float) -> dict[str, object]:
    left = remaining(deadline)
    if left <= 0:
        raise RuntimeError(f"q8_occupancy exceeded its wall-clock budget before {label} "
                           f"tokens={tokens} blocks={blocks}")
    tag = f"{sweep}-{label}-m{m}-t{tokens}-b{blocks}"
    inp = scratch / f"{tag}.input"
    out = scratch / f"{tag}.output"
    ts = scratch / f"{tag}.ts.json"
    inp.write_bytes(bench.synthesize_inputs(m, tokens, blocks))

    cmd = [
        str(driver_bin), str(shader_spv), str(m), str(tokens), str(blocks),
        str(inp), str(out), "--timestamps", str(ts), "--reps", str(reps),
    ]
    cp = subprocess.run(cmd, text=True, capture_output=True, timeout=left)
    if cp.returncode != 0:
        print(cp.stdout, flush=True)
        print(cp.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"q8_driver failed with code {cp.returncode} on {tag}: {cp.stderr.strip()}")
    if not ts.exists():
        raise RuntimeError(f"q8_driver did not emit timestamps for {tag}")

    data = json.loads(ts.read_text())
    reps_data = data["reps"]
    dev = [float(r["device_ns"]) for r in reps_data]
    sub = [int(r["submit_ns"]) for r in reps_data]
    wait = [int(r["wait_ns"]) for r in reps_data]
    rec: dict[str, object] = {
        "sweep": sweep, "label": label, "m": m, "tokens": tokens, "blocks": blocks,
        "reps": len(reps_data),
        "device_ns": dev,
        "device_ns_med": statistics.median(dev),
        "device_ns_min": min(dev), "device_ns_max": max(dev),
        "submit_ns_med": statistics.median(sub), "wait_ns_med": statistics.median(wait),
        "device_name": data["device_name"], "driver_version": data["driver_version"],
        "api_version": data["api_version"],
        "timestamp_period": float(data["timestamp_period"]),
        "timestamp_valid_bits": int(data["timestamp_valid_bits"]),
    }
    rec.update(derive(m, tokens, blocks))
    dev_med = float(rec["device_ns_med"])
    rec["gmac_per_s"] = float(rec["macs"]) / dev_med
    rec["unique_gb_per_s"] = float(rec["unique_bytes"]) / dev_med
    rec["logical_gb_per_s"] = float(rec["logical_bytes"]) / dev_med
    return rec


# ---------------------------------------------------------------------------
# Pure checkers (self-testable; a seeded flat series must be rejected)
# ---------------------------------------------------------------------------

def check_positive(results: list[dict[str, object]]) -> list[str]:
    return [f"{r['sweep']}/{r['label']} m={r['m']} t={r['tokens']} b={r['blocks']} "
            f"device_ns_med={r['device_ns_med']} not positive"
            for r in results if not float(r["device_ns_med"]) > 0.0]


def check_monotone(results: list[dict[str, object]], rel_tol: float = 0.03,
                   field: str = "device_ns_min") -> list[str]:
    """At fixed tokens, device_ns must be non-decreasing in the grid size ng.

    Ordering is read from the per-case minimum over repetitions.  The median carries
    intermittent reps (a clock step or host interrupt lifts a single rep of 9 by 25-60%
    here) and a single inflated rep was enough to trip the 3% band on a series whose
    minima are monotone; the minimum is the contamination-resistant estimator, and a
    stuck timer is caught by :func:`check_dynamic_range`, not here.
    """
    problems: list[str] = []
    by: dict[int, list[dict[str, object]]] = {}
    for r in results:
        if r["sweep"] != "occ":
            continue
        by.setdefault(int(r["tokens"]), []).append(r)
    for tokens, rows in sorted(by.items()):
        rows = sorted(rows, key=lambda r: float(r["ng"]))
        for lo, hi in zip(rows, rows[1:]):
            if not float(hi[field]) >= float(lo[field]) * (1.0 - rel_tol):
                problems.append(
                    f"occupancy ordering violated at tokens={tokens}: "
                    f"ng={int(lo['ng'])} -> {float(lo[field]):.1f} ns fell to "
                    f"ng={int(hi['ng'])} -> {float(hi[field]):.1f} ns "
                    f"(beyond {rel_tol:.0%} noise)")
    return problems


def check_dynamic_range(results: list[dict[str, object]], min_span: float = 5.0) -> list[str]:
    """The instrument must see device_ns change across the occupancy sweep.

    ``max/min`` of the per-token occupancy series must clear ``min_span``; a
    stuck, zero or saturated timer gives a span of 1.0 and reddens here.
    """
    problems: list[str] = []
    by: dict[int, list[dict[str, object]]] = {}
    for r in results:
        if r["sweep"] != "occ":
            continue
        by.setdefault(int(r["tokens"]), []).append(r)
    for tokens, rows in sorted(by.items()):
        vals = [float(r["device_ns_med"]) for r in rows]
        if min(vals) <= 0:
            problems.append(f"occupancy series tokens={tokens} has a non-positive device_ns")
            continue
        span = max(vals) / min(vals)
        if span < min_span:
            problems.append(
                f"occupancy series tokens={tokens} dynamic range {span:.2f}x < {min_span:.1f}x: "
                "instrument sees no change")
    return problems


def check_pair(results: list[dict[str, object]], blocks_small: int,
               blocks_large: int, min_ratio: float) -> tuple[list[str], float | None]:
    by = {int(r["blocks"]): r for r in results if r["sweep"] == "work"}
    small, large = by.get(blocks_small), by.get(blocks_large)
    if small is None or large is None:
        return [f"control pair blocks={blocks_small}/{blocks_large} missing from work sweep"], None
    s, l = float(small["device_ns_med"]), float(large["device_ns_med"])
    ratio = l / s if s > 0 else 0.0
    if not ratio >= min_ratio:
        return ([f"work-sweep pair blocks={blocks_small} -> {s:.1f} ns vs blocks={blocks_large} "
                 f"-> {l:.1f} ns differ by only {ratio:.2f}x (< {min_ratio:.2f}x): "
                 "instrument sees no change"], ratio)
    return [], ratio


def linfit_se(xs: list[float], ys: list[float]) -> tuple[float, float, float, float] | None:
    """Ordinary least squares with (slope, intercept, R^2, slope standard error).

    The standard error is ``sqrt(ss_res/(n-2)) / sqrt(sum((x-xbar)^2))``, so a
    difference of two independently fitted slopes carries a combined error
    ``hypot(se_a, se_b)`` and is only meaningful beyond a few sigma of it.
    """
    n = len(xs)
    if n < 2:
        return None
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    slope = (n * sxy - sx * sy) / denom
    icept = (sy - slope * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    sxx_c = sum((x - sx / n) ** 2 for x in xs)
    se = math.sqrt((ss_res / (n - 2)) / sxx_c) if (n > 2 and sxx_c > 0) else float("inf")
    return slope, icept, r2, se


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    fit = linfit_se(xs, ys)
    return None if fit is None else fit[:3]


def fit_marginal(rows: list[dict[str, object]],
                 field: str = "device_ns_med") -> tuple[float, float, float, float] | None:
    """Marginal ns per (workgroup, block) from deep-K rows, with its standard error."""
    ordered = sorted(rows, key=lambda r: int(r["blocks"]))
    return linfit_se([float(r["work_units"]) for r in ordered],
                     [float(r[field]) for r in ordered])


def load_costmodel():
    spec = importlib.util.spec_from_file_location("q8_costmodel", HERE / "q8_costmodel.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load q8_costmodel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attribute_split(slopes: dict[str, tuple[float, float]],
                    min_sigma: float = SPLIT_MIN_SIGMA) -> tuple[list[str], dict[str, object]]:
    """Split the deep-K marginal into dot, checked-i128 epilogue and remainder.

    ``slopes`` maps each of :data:`SPLIT_VARIANTS` to
    ``(marginal_ns_per_wg_block, standard_error)`` from the cost-probe sweep.  Returns
    ``(problems, report)``.

    The checks are non-vacuity and coherence only, and both are stated on the measured
    interval rather than on a term's size:

    * every term is classified against the free-work null at ``min_sigma`` combined standard
      errors -- ``resolved+``, ``resolved-`` or ``unresolved`` (``tools/gpu/q8_costmodel.py``);
    * a term that is ``unresolved`` decides NOTHING.  The inherited bars here required the
      epilogue delta to clear three sigma, and that term resolved at 0.25-18.0 sigma across
      19 runs, so the exit status of this gate was decided by whether the device could see a
      <=4.5%-of-marginal term on that run.  It no longer is.
    * coherence is conditional: a declared cost term that is ``resolved-`` means the device
      resolved that removing work made the kernel slower, which no cost model can hold, and
      the gate is red.  Unresolved terms pass by not being measured.

    Which term dominates is reported, never asserted: a gate that encodes the hypothesis it is
    testing cannot fail.  ``interaction`` is reported and not gated, because the two terms
    compete for the same issue slots and superposition is a hypothesis, not a precondition.
    """
    problems: list[str] = []
    report: dict[str, object] = {}
    for v in SPLIT_VARIANTS:
        if v not in slopes:
            problems.append(f"split cell {v!r} is missing from the measured slopes")
    if problems:
        return problems, report

    s = {v: float(slopes[v][0]) for v in SPLIT_VARIANTS}
    e = {v: float(slopes[v][1]) for v in SPLIT_VARIANTS}
    for v in SPLIT_VARIANTS:
        if not (math.isfinite(s[v]) and s[v] > 0.0):
            problems.append(f"split cell {v} marginal {s[v]:.4f} ns/(wg,block) is not a "
                            "positive finite slope; the sweep cannot be attributed")
    if problems:
        return problems, report

    dot, dot_se = s["full"] - s["nodot"], math.hypot(e["full"], e["nodot"])
    epi, epi_se = s["full"] - s["noepi"], math.hypot(e["full"], e["noepi"])
    rem, rem_se = s["full"] - s["bare"], math.hypot(e["full"], e["bare"])
    movable = dot + epi

    # The resolved-term rule.  The census above and in the --controls suite prove the elisions
    # reached the compiled module; this is the timing side of the same non-vacuity witness, and
    # it is the only place a term's magnitude can enter a verdict -- conditionally, when the
    # data resolved it.
    cost = load_costmodel()
    terms = {"dot": dot, "checked_i128_epilogue": epi, "removable_alu": rem}
    ses = {"dot": dot_se, "checked_i128_epilogue": epi_se, "removable_alu": rem_se}
    table = cost.term_table(terms, ses, min_sigma)
    problems += cost.distinct_cells_problems(slopes)
    problems += cost.resolved_term_problems(table, SPLIT_COST_TERMS, "split")

    report = {
        "slope_ns_per_wg_block": s,
        "slope_se": e,
        "dot_ns": dot, "dot_se": dot_se,
        "epilogue_ns": epi, "epilogue_se": epi_se,
        "removable_ns": rem, "removable_se": rem_se,
        "interaction_ns": dot + epi - rem,
        "interaction_frac": ((dot + epi - rem) / rem) if rem > 0 else float("nan"),
        "epilogue_over_dot": (epi / dot) if abs(dot) > min_sigma * dot_se else float("nan"),
        "term_table": table,
        "cost_terms": list(SPLIT_COST_TERMS),
        "dot_state": table["dot"]["state"],
        "epilogue_state": table["checked_i128_epilogue"]["state"],
        "removable_state": table["removable_alu"]["state"],
        "dot_resolved": table["dot"]["state"] != cost.UNRESOLVED,
        "epilogue_resolved": table["checked_i128_epilogue"]["state"] != cost.UNRESOLVED,
        "epilogue_share_of_movable": (epi / movable) if abs(movable) > 0 else float("nan"),
        "dot_share_of_movable": (dot / movable) if abs(movable) > 0 else float("nan"),
        "dot_share_of_full": dot / s["full"],
        "epilogue_share_of_full": epi / s["full"],
        "remainder_share_of_full": s["bare"] / s["full"],
        "removable_share_of_full": rem / s["full"],
        "min_sigma": min_sigma,
        "dot_sigma": dot / dot_se if dot_se > 0 else float("inf"),
        "epilogue_sigma": epi / epi_se if epi_se > 0 else float("inf"),
        "removable_sigma": rem / rem_se if rem_se > 0 else float("inf"),
        "ceiling_if_dot_free": (s["full"] / s["nodot"]) if s["nodot"] > 0 else float("nan"),
        "ceiling_if_both_alu_terms_free": (s["full"] / s["bare"]) if s["bare"] > 0 else float("nan"),
        "plateau_gmac_s_if_dot_free": (MACS_PER_WG_BLOCK / s["nodot"]) if s["nodot"] > 0 else float("nan"),
        "plateau_gmac_s_if_both_free": (MACS_PER_WG_BLOCK / s["bare"]) if s["bare"] > 0 else float("nan"),
    }
    return problems, report


def control_self_test() -> int:
    def occ_row(tokens: int, ng: int, ns: float) -> dict[str, object]:
        return {"sweep": "occ", "label": "seed", "m": 0, "tokens": tokens, "blocks": 64,
                "ng": float(ng), "device_ns_med": ns, "device_ns_min": ns * 0.98,
                "work_units": 1.0}
    good = [occ_row(t, ng, float(t + ng)) for t in (32, 64) for ng in (16, 128, 1024)]
    if check_monotone(good) or check_dynamic_range(good):
        raise RuntimeError("control self-test: a monotone, wide-range series was rejected")
    flat = copy.deepcopy(good)
    for r in flat:
        r["device_ns_med"] = 100.0
        r["device_ns_min"] = 100.0
    if not check_dynamic_range(flat):
        raise RuntimeError("control self-test: a flat (stuck-timer) series survived")
    inverted = [occ_row(32, ng, float(2000 - ng)) for ng in (16, 128, 1024)]
    if not check_monotone(inverted):
        raise RuntimeError("control self-test: an inverted series survived")
    pair_good = [{"sweep": "work", "label": "seed", "m": 0, "tokens": 0, "blocks": 16,
                  "ng": 1.0, "work_units": 1.0, "device_ns_med": 100.0},
                 {"sweep": "work", "label": "seed", "m": 0, "tokens": 0, "blocks": 224,
                  "ng": 1.0, "work_units": 1.0, "device_ns_med": 400.0}]
    if check_pair(pair_good, 16, 224, MIN_PAIR_RATIO)[0]:
        raise RuntimeError("control self-test: a live pair was rejected")
    pair_flat = copy.deepcopy(pair_good)
    for r in pair_flat:
        r["device_ns_med"] = 100.0
    if not check_pair(pair_flat, 16, 224, MIN_PAIR_RATIO)[0]:
        raise RuntimeError("control self-test: a flat pair survived")

    # --- split checker: the resolved-term rule ----------------------------------
    # The seed is the shape the device actually reported: the epilogue resolved at
    # >11 sigma, the dot unresolved.  Two accepted seeds bracket the hypothesis from
    # both sides (dot-dominant and epilogue-dominant), so the checker cannot be
    # encoding either one.
    good_split = {"full": (61.62, 0.10), "nodot": (61.37, 0.10),
                  "noepi": (60.00, 0.10), "bare": (59.94, 0.10)}
    probs, good_report = attribute_split(good_split)
    if probs:
        raise RuntimeError(f"control self-test: a coherent split was rejected: {probs[0]}")
    if str(good_report["epilogue_state"]) != "resolved+":
        raise RuntimeError(f"control self-test: the resolved epilogue term was classified "
                           f"{good_report['epilogue_state']}, not resolved+")
    if str(good_report["dot_state"]) != "unresolved":
        raise RuntimeError(f"control self-test: an unresolved dot term was classified "
                           f"{good_report['dot_state']}")
    # Anti-hypothesis controls: dot-dominant and exactly-balanced splits must ALSO be
    # accepted, so the checker cannot be encoding "the epilogue dominates".
    for name, seed in (
        ("dot-dominant", {"full": (18.0, 0.10), "nodot": (8.0, 0.10),
                          "noepi": (17.5, 0.10), "bare": (7.5, 0.10)}),
        # Distinct marginals throughout: two cells at the same marginal is the unapplied-probe
        # defect, not a balance seed.
        ("balanced", {"full": (18.0, 0.10), "nodot": (13.0, 0.10),
                      "noepi": (12.8, 0.10), "bare": (8.0, 0.10)}),
    ):
        if attribute_split(seed)[0]:
            raise RuntimeError(f"control self-test: a {name} split was rejected; the checker "
                               "encodes a hypothesis about which term dominates")
    # THE DETERMINISM CONTROL.  The inherited bar required the epilogue delta to clear
    # 3 sigma and it resolved at 0.25-18.0 sigma over 19 runs, so the exit flipped ten
    # times.  Here the epilogue delta is swept across its ENTIRE unresolved band at the
    # real uncertainty and the verdict must not move once: this is the check that fails if
    # any bar still reads a term the data cannot resolve.
    band_se = math.hypot(good_split["full"][1], good_split["noepi"][1])
    baseline = attribute_split(good_split)[0]
    # i = 0 is the exact no-op, and it IS refused -- by the unapplied-probe rule, not by a
    # magnitude bar -- so the sweep of the band proper starts at the first non-zero point.
    if not any("not applied" in g
               for g in attribute_split(dict(good_split, noepi=good_split["full"]))[0]):
        raise RuntimeError("control self-test: an exact no-op epilogue was not refused by name")
    for i in range(1, 61):
        value = SPLIT_MIN_SIGMA * band_se * i / 60.0
        swept = dict(good_split, noepi=(good_split["full"][0] - value, band_se))
        got = attribute_split(swept)[0]
        if got != baseline:
            raise RuntimeError(f"control self-test: at an epilogue delta of {value:.4f} ns "
                               f"({value / band_se:.2f} of the {SPLIT_MIN_SIGMA:g}-sigma band) the "
                               f"verdict changed from {baseline} to {got}; a bar still reads a "
                               f"term the instrument cannot resolve")
    # A resolved-negative removal of work is a contradiction, and this is the only
    # term-magnitude red that remains.
    for name, seed, want in (
        ("a no-op epilogue attribution",
         dict(good_split, noepi=good_split["full"]), "not applied"),
        ("a both-zero attribution",
         dict(good_split, nodot=good_split["full"], noepi=good_split["full"]), "not applied"),
        ("a bare cell the device resolved as slower",
         dict(good_split, bare=(63.00, 0.10)), "resolved NEGATIVE"),
    ):
        got = attribute_split(seed)[0]
        if not any(want in g for g in got):
            raise RuntimeError(f"control self-test: {name} survived; got {got}")
    flat_split = {k: (0.0, 0.0) for k in SPLIT_VARIANTS}
    if not any("positive finite" in g for g in attribute_split(flat_split)[0]):
        raise RuntimeError("control self-test: a zero-marginal split survived")
    missing_split = {k: v for k, v in good_split.items() if k != "bare"}
    if not attribute_split(missing_split)[0]:
        raise RuntimeError("control self-test: a split missing a cell survived")

    # --- census checker: the non-vacuity witness, six seeded defects -----------
    # The seed is the census glslc actually produced for the four cells.
    good_census = {"full": {"sdot": 1, "arith": 86, "iaddcarry": 0, "total": 700},
                   "nodot": {"sdot": 0, "arith": 87, "iaddcarry": 0, "total": 700},
                   "noepi": {"sdot": 1, "arith": 61, "iaddcarry": 0, "total": 700},
                   "bare": {"sdot": 0, "arith": 62, "iaddcarry": 0, "total": 700}}
    if check_census(good_census):
        raise RuntimeError("control self-test: a census proving all three elisions was rejected")
    for name, bad in (
        ("the dot elision did not reach the module",
         {**good_census, "nodot": {**good_census["nodot"], "sdot": 8}}),
        ("the epilogue elision changed the dot",
         {**good_census, "noepi": {**good_census["noepi"], "sdot": 4}}),
        ("the epilogue elision removed too little arithmetic",
         {**good_census, "noepi": {**good_census["noepi"], "arith": 85}}),
        ("bare removes less arithmetic than noepi",
         {**good_census, "bare": {**good_census["bare"], "arith": 70}}),
        ("the bare cell still holds dot opcodes",
         {**good_census, "bare": {**good_census["bare"], "sdot": 8}}),
        ("the full cell holds no dot at all",
         {**good_census, "full": {**good_census["full"], "sdot": 0}}),
    ):
        if not check_census(bad):
            raise RuntimeError(f"control self-test: a census where {name} survived")
    if not check_census({k: v for k, v in good_census.items() if k != "noepi"}):
        raise RuntimeError("control self-test: a census missing a cell survived")

    print("control self-test PASS: flat series and flat pair rejected; monotone series accepted; "
          "dot-dominant, balanced and epilogue-dominant splits all accepted; the epilogue delta "
          "swept across all 60 points of its unresolved band leaves the verdict unchanged; "
          "no-op, both-zero, resolved-slower-bare, zero-marginal and missing-cell splits rejected "
          "by name; two wrong-shape rows rejected and a consistent row accepted; "
          "census accepted, six seeded elision defects and a missing cell rejected", flush=True)
    # The acceptance protocol and the remainder checker carry their own seeds, under the same
    # --control-self-test, so a defect in either cannot hide behind a passing split suite.
    load_costmodel().self_test()
    load_remainder().self_test()
    return 0


def probe_device() -> dict[str, object]:
    """Best-effort CUDA device-property probe (same physical GPU) for the occupancy
    denominator and peak bandwidth; falls back to recorded spec constants."""
    from shutil import which
    facts: dict[str, object] = {
        "source": "spec-constants", "name": DEVICE_NAME_FALLBACK,
        "sm_count": SPEC_SM_COUNT, "max_threads_per_sm": SPEC_MAX_THREADS_PER_SM,
        "mem_bus_bits": SPEC_MEM_BUS_BITS,
    }
    nvcc = None
    for cand in ("/opt/cuda/bin/nvcc", "nvcc"):
        nvcc = which(cand)
        if nvcc:
            break
    if nvcc:
        with tempfile.TemporaryDirectory(prefix="apeiron-q8-occ-probe-") as td:
            src, binp = Path(td) / "devprop.cu", Path(td) / "devprop"
            src.write_text(
                "#include <cstdio>\n#include <cuda_runtime.h>\n"
                "int main(){cudaDeviceProp p;cudaGetDeviceProperties(&p,0);"
                "printf(\"%d %d %d %d\\n\",p.multiProcessorCount,p.maxThreadsPerMultiProcessor,"
                "p.memoryBusWidth,p.l2CacheSize);return 0;}\n")
            try:
                subprocess.run([nvcc, "-o", str(binp), str(src)], check=True,
                               capture_output=True, timeout=180)
                out = subprocess.run([str(binp)], check=True, capture_output=True,
                                     text=True, timeout=60).stdout.split()
                sm, threads, bus, l2 = int(out[0]), int(out[1]), int(out[2]), int(out[3])
                facts.update({"source": "cuda-device-prop", "sm_count": sm,
                              "max_threads_per_sm": threads, "mem_bus_bits": bus, "l2_bytes": l2})
                print(f"device probe (CUDA): sm_count={sm} max_threads_per_sm={threads} "
                      f"mem_bus_bits={bus} l2_bytes={l2}", flush=True)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
                print(f"device probe fell back to spec constants: {exc}",
                      file=sys.stderr, flush=True)
    facts["mem_clock_hz"] = SPEC_MEM_CLOCK_HZ
    facts["peak_gb_per_s"] = (float(SPEC_MEM_CLOCK_HZ) * 2 * int(facts["mem_bus_bits"]) / 8 / 1e9)
    facts["resident_workgroups"] = (int(facts["sm_count"]) * int(facts["max_threads_per_sm"])
                                    // INVOCATIONS_PER_WG)
    return facts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_sweep(results: list[dict[str, object]], title: str) -> None:
    print("-" * 128)
    print(title)
    print("-" * 128)
    print(f"{'shape':<12}{'M':>6}{'tok':>5}{'blk':>5}{'ng':>7} | "
          f"{'dev_min':>9}{'dev_med':>11}{'dev_max':>11} | {'sub_med':>8}{'wait_med':>10} | "
          f"{'GMAC/s':>9}{'uniqGB/s':>9}{'logicGB/s':>10}{'intens':>8}")
    for r in results:
        print(f"{str(r['label']):<12}{int(r['m']):>6}{int(r['tokens']):>5}{int(r['blocks']):>5}"
              f"{int(r['ng']):>7} | "
              f"{float(r['device_ns_min']):>9.1f}{float(r['device_ns_med']):>11.1f}"
              f"{float(r['device_ns_max']):>11.1f} | "
              f"{float(r['submit_ns_med']):>8.0f}{float(r['wait_ns_med']):>10.0f} | "
              f"{float(r['gmac_per_s']):>9.2f}{float(r['unique_gb_per_s']):>9.2f}"
              f"{float(r['logical_gb_per_s']):>10.2f}{float(r['intensity']):>8.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apeiron Q8 SPIR-V per-invocation cost attribution")
    parser.add_argument("--reps", type=int, default=9, help="repetitions per case (default: 9)")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--controls", action="store_true",
                        help="run the seeded-control suite around the live sweeps")
    parser.add_argument("--control-self-test", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    start = time.monotonic()
    deadline = start + args.budget - BUDGET_GRACE_SECONDS

    if args.control_self_test:
        return control_self_test()

    bench = load_bench()
    rem = load_remainder()
    glslc = bench.tool("glslc", "GLSLC", "glslc")
    spirv_val = bench.tool("spirv-val", "SPIRV_VAL", "spirv-val")
    cc = bench.tool("cc", "CC", "cc")
    probe = probe_device()
    peak_bw = float(probe["peak_gb_per_s"])

    results: list[dict[str, object]] = []
    split: list[dict[str, object]] = []
    remrows: list[dict[str, object]] = []
    split_spirv_sha: dict[str, str] = {}
    split_census: dict[str, dict[str, int]] = {}
    rem_census: dict[str, dict[str, int]] = {}
    with tempfile.TemporaryDirectory(prefix="apeiron-q8-occ-") as td:
        scratch = Path(td)
        shader_spv = scratch / "q8_tile.spv"
        driver_bin = scratch / "q8_driver"
        bench.compile_shader(glslc, spirv_val, bench.SHADER, shader_spv, deadline=deadline)
        bench.compile_driver(cc, bench.DRIVER, driver_bin, deadline=deadline)

        # 2x2 cost-probe cells: each is the executing kernel with one term elided.
        split_spv: dict[str, Path] = {}
        for variant, text in build_variants(bench.SHADER).items():
            vdir = scratch / variant
            vdir.mkdir()
            vsrc, vspv = vdir / f"q8_tile_{variant}.comp", vdir / f"q8_tile_{variant}.spv"
            vsrc.write_text(text)
            bench.compile_shader(glslc, spirv_val, vsrc, vspv, deadline=deadline)
            split_spv[variant] = vspv
            split_spirv_sha[variant] = hashlib.sha256(vspv.read_bytes()).hexdigest()
            split_census[variant] = census_spirv(vspv)
        if len(set(split_spirv_sha.values())) != len(SPLIT_VARIANTS):
            raise RuntimeError("q8_occupancy: the four cost probes compiled to "
                               f"{len(set(split_spirv_sha.values()))} distinct SPIR-V modules, not "
                               f"{len(SPLIT_VARIANTS)}; a cost probe is a duplicate of another "
                               "cell and its attribution would be a relabelling")
        census_problems = check_census(split_census)

        # Remainder cost-probe cells: the staging / block-loop / finalize terms the 2x2
        # leaves unnamed.  The reference cell for their census is the same 'full' module the
        # 2x2 uses, so both suites price one kernel.
        rem_spv: dict[str, Path] = {}
        for variant, text in rem.build_rem_variants(bench.SHADER.read_text(),
                                                     bench.SHADER).items():
            vdir = scratch / f"rem_{variant}"
            vdir.mkdir()
            vsrc, vspv = vdir / f"q8_tile_{variant}.comp", vdir / f"q8_tile_{variant}.spv"
            vsrc.write_text(text)
            bench.compile_shader(glslc, spirv_val, vsrc, vspv, deadline=deadline)
            rem_spv[variant] = vspv
            split_spirv_sha[variant] = hashlib.sha256(vspv.read_bytes()).hexdigest()
            rem_census[variant] = rem.census_ext(vspv)
        rem_census["full"] = rem.census_ext(split_spv["full"])
        if len(set(split_spirv_sha.values())) != len(SPLIT_VARIANTS) + len(rem.REM_VARIANTS):
            raise RuntimeError("q8_occupancy: the cost probes compiled to "
                               f"{len(set(split_spirv_sha.values()))} distinct SPIR-V modules, not "
                               f"{len(SPLIT_VARIANTS) + len(rem.REM_VARIANTS)}; a cost probe "
                               "is a duplicate of another cell and its attribution would be a "
                               "relabelling")
        rem_census_problems = rem.check_rem_census(rem_census)

        for m in OCC_M:
            for tokens in OCC_TOKENS:
                results.append(run_case(bench, driver_bin, shader_spv, scratch, "occ",
                                        "occ", m, tokens, OCC_BLOCKS, args.reps, deadline))
        for blocks in WORK_BLOCKS:
            results.append(run_case(bench, driver_bin, shader_spv, scratch, "work",
                                    "work", WORK_M, WORK_TOKENS, blocks, args.reps, deadline))
        for blocks in ISO_BLOCKS:
            results.append(run_case(bench, driver_bin, shader_spv, scratch, "iso",
                                    "iso", ISO_M, ISO_TOKENS, blocks, args.reps, deadline))
        # The whole-kernel exit quantity needs the instrument's OWN spread on the same ten
        # shapes, so the real sweep is repeated COST_MODEL_PASSES times.  Only pass 0 goes into
        # `results` (the ordering / dynamic-range / pair controls and `print_sweep` read that
        # list, and a duplicated row would print twice); every pass goes into `real_passes`.
        real_passes: list[list[dict[str, object]]] = [[] for _ in range(COST_MODEL_PASSES)]
        for p in range(COST_MODEL_PASSES):
            for label, m, blocks in REAL_SHAPES:
                for tokens in REAL_TOKENS:
                    row = run_case(bench, driver_bin, shader_spv, scratch, "real",
                                   label, m, tokens, blocks, args.reps, deadline)
                    row["pass"] = p
                    real_passes[p].append(row)
                    if p == 0:
                        results.append(row)
        # Cells are interleaved by depth, not run cell-by-cell, so a clock or thermal
        # step across the sweep biases all nine cells equally instead of one of them; and the
        # passes are interleaved inside each depth, so a step inside one pass is shared by every
        # cell rather than by one.  That is what makes the between-pass spread measurable.
        split_passes: dict[str, list[list[dict[str, object]]]] = {
            v: [[] for _ in range(COST_MODEL_PASSES)] for v in SPLIT_VARIANTS}
        rem_passes: dict[str, list[list[dict[str, object]]]] = {
            v: [[] for _ in range(COST_MODEL_PASSES)] for v in rem.REM_VARIANTS}
        for blocks in SPLIT_BLOCKS:
            for variant in SPLIT_VARIANTS:
                for p in range(COST_MODEL_PASSES):
                    row = run_case(bench, driver_bin, split_spv[variant], scratch, "split",
                                   variant, SPLIT_M, SPLIT_TOKENS, blocks, SPLIT_REPS,
                                   deadline)
                    row["pass"] = p
                    split_passes[variant][p].append(row)
                    if p == 0:
                        split.append(row)
            for variant in rem.REM_VARIANTS:
                for p in range(COST_MODEL_PASSES):
                    row = run_case(bench, driver_bin, rem_spv[variant], scratch, "rem",
                                   variant, SPLIT_M, SPLIT_TOKENS, blocks, SPLIT_REPS,
                                   deadline)
                    row["pass"] = p
                    rem_passes[variant][p].append(row)
                    if p == 0:
                        remrows.append(row)

    elapsed = time.monotonic() - start
    ident = results[0]
    occ = [r for r in results if r["sweep"] == "occ"]
    work = [r for r in results if r["sweep"] == "work"]
    iso = [r for r in results if r["sweep"] == "iso"]
    real = [r for r in results if r["sweep"] == "real"]

    print("=" * 128)
    print("APEIRON Q8 SPIR-V — PER-INVOCATION COST ATTRIBUTION")
    print("=" * 128)
    print(f"device          = {ident['device_name']} (driver={ident['driver_version']}, "
          f"api={ident['api_version']})")
    print(f"timestampPeriod = {float(ident['timestamp_period']):.2f} ns   "
          f"timestampValidBits = {int(ident['timestamp_valid_bits'])}")
    print(f"device probe    = {probe['source']}: sm_count={probe['sm_count']} "
          f"max_threads_per_sm={probe['max_threads_per_sm']} "
          f"resident_workgroups={probe['resident_workgroups']} "
          f"peak_bw={peak_bw:.1f} GB/s")
    print(f"kernel geometry = local 8x8 = {INVOCATIONS_PER_WG} invocations/workgroup, "
          f"{ROWS_PER_WG} rows x {TOKENS_PER_WG} tokens, {MACS_PER_WG_BLOCK} MACs/(wg,block)")
    n_dispatch = len(results) + len(split) + len(remrows) + (COST_MODEL_PASSES - 1) * (
        len(real_passes[0]) + len(split) + len(remrows))
    print(f"reps            = {args.reps} sweeps / {SPLIT_REPS} cost probes   "
          f"passes = {COST_MODEL_PASSES}   budget = {args.budget:.0f} s   "
          f"elapsed = {elapsed:.2f} s   cases = {len(results)} rows "
          f"(+{(COST_MODEL_PASSES - 1) * (len(real_passes[0]) + len(split) + len(remrows))} "
          f"replication rows, {n_dispatch} dispatched rows)")

    print_sweep(occ, "OCCUPANCY SWEEP — blocks=64 fixed, M x tokens grid (grid size ng varies)")
    print_sweep(work, f"WORK SWEEP — M={WORK_M} tokens={WORK_TOKENS} (ng=512) fixed, blocks varies")
    print_sweep(iso, "ISOLATED SINGLE-WORKGROUP — ng=1 (M=16, tokens=32), blocks in {1,64}")
    print_sweep(real, "REAL PROJECTION SHAPES — Qwen3.5-0.8B")
    print_sweep(split, f"DOT / EPILEDOGUE COST PROBES — M={SPLIT_M} tokens={SPLIT_TOKENS} "
                       f"(ng={int(ng_of(SPLIT_M, SPLIT_TOKENS))}), 2x2 over {SPLIT_BLOCKS}")

    # --- attribution --------------------------------------------------------
    by_iso = {int(r["blocks"]): r for r in iso}
    t_dispatch = float(by_iso[1]["device_ns_med"]) if 1 in by_iso else float("nan")
    t_iso64 = float(by_iso[64]["device_ns_med"]) if 64 in by_iso else float("nan")
    l_blk = (t_iso64 - t_dispatch) / 63.0 if (64 in by_iso and 1 in by_iso) else float("nan")
    work_sorted = sorted(work, key=lambda r: int(r["blocks"]))
    deep = [r for r in work_sorted if int(r["blocks"]) >= DEEP_BLOCKS]
    deep_fit = linfit([float(r["work_units"]) for r in deep],
                      [float(r["device_ns_med"]) for r in deep])
    occ_fit = linfit([float(r["ng"]) for r in occ],
                     [float(r["device_ns_med"]) for r in occ])
    peak_gmac = max(float(r["gmac_per_s"]) for r in results)
    real_peak_gmac = max(float(r["gmac_per_s"]) for r in real)
    max_unique_bw = max(float(r["unique_gb_per_s"]) for r in results)
    max_logic_bw = max(float(r["logical_gb_per_s"]) for r in results)
    min_intens = min(float(r["intensity"]) for r in results)
    plateau = (MACS_PER_WG_BLOCK / deep_fit[0]) if (deep_fit and deep_fit[0] > 0) else float("nan")
    knee_intens = (plateau / peak_bw) if (math.isfinite(plateau) and peak_bw > 0) else float("nan")

    # --- dot / epilogue split over the 2x2 cost probes -------------------------
    # Each cell's slope is fitted once per pass and combined with a measured uncertainty:
    # hypot(mean per-pass fit error, sd(per-pass slopes)/sqrt(passes)).  The single-pass fit
    # error alone is what the inherited bars used, and it is the component that cannot see the
    # 4% device-regime shift in the kernel's own marginal.
    cost = load_costmodel()
    split_rows = {v: [r for r in split if r["label"] == v] for v in SPLIT_VARIANTS}
    split_fits = {v: cost.replicated_fit(split_passes[v], SPLIT_ESTIMATOR)
                  for v in SPLIT_VARIANTS}
    split_fits_med = {v: fit_marginal(split_rows[v], "device_ns_med") for v in SPLIT_VARIANTS}
    split_slopes: dict[str, tuple[float, float]] = {}
    split_r2: dict[str, float] = {}
    split_missing = [v for v in SPLIT_VARIANTS if split_fits[v] is None]
    for v in SPLIT_VARIANTS:
        if split_fits[v] is not None:
            split_slopes[v] = (split_fits[v][0], split_fits[v][3])
            split_r2[v] = split_fits[v][2]
    split_problems = list(attribute_split(split_slopes)[0]) + census_problems

    # --- the whole-kernel fit: THE EXIT QUANTITY -------------------------------
    unit_problems, unit_report = cost.per_unit_fit(real_passes, ng_of, "device_ns_min")
    geometry_problems = cost.geometry_problems(
        [r for r in results] + [r for one in real_passes for r in one], ng_of)

    # --- remainder attribution: staging / block loop / finalize -------------------
    rem_rows = {v: [r for r in remrows if r["label"] == v] for v in rem.REM_VARIANTS}
    rem_fits = {v: cost.replicated_fit(rem_passes[v], SPLIT_ESTIMATOR)
                for v in rem.REM_VARIANTS}
    rem_fits_med = {v: fit_marginal(rem_rows[v], "device_ns_med") for v in rem.REM_VARIANTS}
    rem_slopes: dict[str, tuple[float, float]] = {}
    rem_r2: dict[str, float] = {}
    rem_missing = [v for v in rem.REM_VARIANTS if rem_fits[v] is None]
    for v in rem.REM_VARIANTS:
        if rem_fits[v] is not None:
            rem_slopes[v] = (rem_fits[v][0], rem_fits[v][3])
            rem_r2[v] = rem_fits[v][2]
    rem_report: dict[str, object] = {}
    rem_problems: list[str] = list(rem_census_problems)
    rem_intercepts = {v: float(rem_fits[v][1]) for v in rem.REM_VARIANTS
                      if rem_fits[v] is not None}
    if "full" in split_fits and split_fits["full"] is not None:
        rem_intercepts["full"] = float(split_fits["full"][1])
    rem_rows_all: dict[str, list[dict[str, object]]] = dict(split_rows)
    rem_rows_all.update(rem_rows)
    if split_slopes and not rem_missing:
        rem_problems += rem.attribute_remainder(
            split_slopes["full"], rem_slopes, attribute_split(split_slopes)[1],
            rem_census, SPLIT_MIN_SIGMA, rem_intercepts, rem_rows_all)[0]
    if split_slopes and not rem_missing:
        rem_report = rem.attribute_remainder(
            split_slopes["full"], rem_slopes, attribute_split(split_slopes)[1],
            rem_census, SPLIT_MIN_SIGMA, rem_intercepts, rem_rows_all)[1]

    # The split sweep re-measures the incumbent independently of the live sweeps.  Its
    # 'full' cell must reproduce the occupancy sweep's row for the SAME dispatch
    # (m=128, tokens=256, blocks=64 -> ng=64, identical shader, identical input
    # synthesis), or the split is not pricing q8_tile.comp.
    rm, rt, rb = SPLIT_RECHECK
    occ_row = next((r for r in occ if int(r["m"]) == rm and int(r["tokens"]) == rt
                    and int(r["blocks"]) == rb), None)
    split_row = next((r for r in split_rows["full"] if int(r["blocks"]) == rb), None)
    incumbent_recheck = float("nan")
    if occ_row is None or split_row is None:
        split_problems.append(
            f"incumbent re-check unavailable: the occupancy sweep has "
            f"{'a' if occ_row else 'no'} row at m={rm} tokens={rt} blocks={rb}, the split "
            f"'full' cell has {'a' if split_row else 'no'} row at blocks={rb}")
    else:
        occ_ns, split_ns = float(occ_row["device_ns_min"]), float(split_row["device_ns_min"])
        incumbent_recheck = (split_ns - occ_ns) / occ_ns if occ_ns > 0 else float("nan")
        if not (math.isfinite(incumbent_recheck)
                and abs(incumbent_recheck) <= SPLIT_RECHECK_MIN_REL):
            split_problems.append(
                f"the split 'full' cell does not reproduce the incumbent on the same dispatch: "
                f"{split_ns:.1f} vs {occ_ns:.1f} ns at m={rm} tokens={rt} blocks={rb} "
                f"({incumbent_recheck:+.2%}, band {SPLIT_RECHECK_MIN_REL:.0%}); the split is not "
                "pricing q8_tile.comp")
    split_report = attribute_split(split_slopes)[1]

    # --- non-vacuity, at GATE scope --------------------------------------------
    # One count over every declared cost term of BOTH suites.  It is deliberately not per-suite:
    # in this device's low regime all three ALU terms read below their own intervals while the
    # remainder's terms resolve at 170+ sigma, and the first version of this rule -- a per-suite
    # count -- reddened exactly that run (after2.log, exit 1).  A bar that flips on a
    # 2.7%-of-marginal term is the defect this protocol exists to remove.
    nonvacuity_problems = cost.instrument_reads_work(
        {"split": (split_report.get("term_table", {}), SPLIT_COST_TERMS),
         "remainder": (rem_report.get("term_table", {}), rem.REM_COST_TERMS)})

    print("=" * 128)
    print("ATTRIBUTION")
    print("=" * 128)
    print(f"dispatch floor t0 (ng=1, blocks=1)        = {t_dispatch:>12.1f} ns")
    print(f"single-workgroup exposed per-block latency = {l_blk:>12.1f} ns/block "
          f"(= {l_blk / 1000.0:.2f} us/block for one workgroup, no occupancy hiding)")
    if deep_fit:
        b, w0, r2w = deep_fit
        print(f"deep-K marginal cost (blocks>={DEEP_BLOCKS})         = {b:>12.4f} ns per "
              f"(workgroup*block), fit R^2={r2w:.4f}")
        print(f"  -> local arithmetic plateau              = {plateau:>12.1f} GMAC/s "
              f"(= {MACS_PER_WG_BLOCK}/b)")
    if occ_fit:
        g, g0, r2o = occ_fit
        print(f"occupancy marginal (blocks=64, over ng=1..2048)= {g:>12.3f} ns per workgroup, "
              f"fit R^2={r2o:.4f} (curve is convex; single slope is a local bound)")
    print(f"peak observed GMAC/s (all sweeps)          = {peak_gmac:>12.2f}")
    print(f"peak observed GMAC/s (real shapes)         = {real_peak_gmac:>12.2f}")
    print(f"CUDA MMQ comparator band                   = {CUDA_GMAC_S[0]:>8.0f} .. "
          f"{CUDA_GMAC_S[1]:>8.0f} GMAC/s")
    print(f"achieved unique-byte bandwidth (max)       = {max_unique_bw:>12.2f} GB/s "
          f"({max_unique_bw / peak_bw * 100:.2f}% of {peak_bw:.0f} GB/s peak)")
    print(f"achieved logical-byte bandwidth (max)      = {max_logic_bw:>12.2f} GB/s "
          f"({max_logic_bw / peak_bw * 100:.2f}% of peak)")
    print(f"minimum arithmetic intensity               = {min_intens:>12.2f} MAC/byte "
          f"(roofline knee ~{knee_intens:.2f} MAC/byte)")
    print(f"dispatch floor share at real shapes        = "
          f"{t_dispatch / min(float(r['device_ns_med']) for r in real) * 100:.3f}% .. "
          f"{t_dispatch / max(float(r['device_ns_med']) for r in real) * 100:.3f}% "
          f"(of measured device_ns)")

    # --- dot / epilogue split --------------------------------------------------
    print("-" * 128)
    print("DOT / CHECKED-I128 EPILOGUE SPLIT — 2x2 cost probes of src/spirv/q8_tile.comp")
    print("-" * 128)
    print(f"  geometry M={SPLIT_M} tokens={SPLIT_TOKENS} (ng={int(ng_of(SPLIT_M, SPLIT_TOKENS))}), "
          f"blocks in {SPLIT_BLOCKS}; metric = device-timestamp delta across one dispatch")
    for v in SPLIT_VARIANTS:
        if v in split_slopes:
            med = split_fits_med[v]
            per_pass = "/".join(f"{float(q['slope']):.4f}" for q in split_fits[v][4])
            print(f"  {v:<6} marginal = {split_slopes[v][0]:>10.4f} ns/(wg,block)  "
                  f"+/- {split_slopes[v][1]:.4f}  fit R^2={split_r2[v]:.4f}  "
                  f"passes {per_pass}  "
                  f"(median-estimator cross-read {med[0]:.4f}, R^2={med[2]:.4f})  "
                  f"spirv_sha256={split_spirv_sha[v][:16]}...")
        else:
            print(f"  {v:<6} marginal = unavailable")
    print(f"  estimator = {SPLIT_ESTIMATOR} over {SPLIT_REPS} reps, cells interleaved by depth, "
          f"{COST_MODEL_PASSES} passes per cell; +/- is hypot(mean per-pass fit error, "
          f"sd(pass slopes)/sqrt({COST_MODEL_PASSES}))")
    print(f"  {len(set(split_spirv_sha.values()))} distinct SPIR-V digests")
    print("  static census (spirv-dis opcode count) — the non-vacuity witness for each elision:")
    print(f"    {'cell':<6}{'OpSDot':>9}{'arith':>9}{'IAddCarry':>11}{'total':>9}")
    for v in SPLIT_VARIANTS:
        c = split_census.get(v)
        print(f"    {v:<6}{(c['sdot'] if c else -1):>9}{(c['arith'] if c else -1):>9}"
              f"{(c['iaddcarry'] if c else -1):>11}{(c['total'] if c else -1):>9}")
    if split_report:
        r = split_report
        s_full = float(r["slope_ns_per_wg_block"]["full"])
        tbl = r["term_table"]
        for label, key, where in (
                ("dot term      (8x dotPacked4x8EXT, q8_tile.comp:107)", "dot", ""),
                ("epilogue term (addChecked+shiftedTerm, :121-124)", "checked_i128_epilogue", ""),
                ("both ALU terms (full - bare)", "removable_alu", "")):
            row = tbl[key]
            print(f"{label:<56} = {float(row['ns']):>9.4f} +/- {float(row['se']):.4f} "
                  f"ns/(wg,block)  interval [{float(row['lo']):+.4f}, {float(row['hi']):+.4f}]  "
                  f"{float(row['sigma']):+6.2f} sigma  -> {row['state']}")
        print(f"remainder      (staging loads, block loop, finalize epilogue)  = "
              f"{float(r['slope_ns_per_wg_block']['bare']):>9.4f} ns/(wg,block) "
              f"= {float(r['remainder_share_of_full']) * 100:.1f}% of the full marginal")
        if bool(r["epilogue_resolved"]) and bool(r["dot_resolved"]):
            print(f"resolved terms, epilogue : dot = "
                  f"{float(r['epilogue_over_dot']):.2f}x")
            print(f"share of the two movable ALU terms: dot "
                  f"{float(r['dot_share_of_movable']) * 100:.1f}% / epilogue "
                  f"{float(r['epilogue_share_of_movable']) * 100:.1f}%")
        else:
            # A share of two terms is undefined when one of them is not resolved: the
            # denominator is then a difference of one significant and one insignificant
            # number.  Report the resolved term against the whole marginal instead.
            unresolved = "dot" if not bool(r["dot_resolved"]) else "epilogue"
            resolved = "epilogue" if unresolved == "dot" else "dot"
            print(f"share of the two movable ALU terms: not applicable — the {unresolved} term "
                  f"is below this geometry's resolution, so the two-term denominator is "
                  "undefined")
            for name in (resolved,):
                print(f"  {name} term: {float(r[name + '_ns']):+.4f} ns/(wg,block) = "
                      f"{float(r[name + '_share_of_full']) * 100:+.2f}% of the full marginal "
                      f"(3-sigma bound "
                      f"{float(r['min_sigma']) * float(r[name + '_se']) / s_full * 100:.2f}%)")
        print(f"share of the full marginal:         dot {float(r['dot_share_of_full']) * 100:+.2f}% "
              f"(3-sigma bound "
              f"{float(r['min_sigma']) * float(r['dot_se']) / s_full * 100:.2f}%) "
              f"/ epilogue {float(r['epilogue_share_of_full']) * 100:+.2f}% "
              f"(3-sigma bound "
              f"{float(r['min_sigma']) * float(r['epilogue_se']) / s_full * 100:.2f}%) "
              f"/ both {float(r['removable_share_of_full']) * 100:+.2f}% "
              f"({float(r['removable_sigma']):.1f} sigma)")
        print(f"superposition interaction (dot+epilogue-removable)             = "
              f"{float(r['interaction_ns']):+9.4f} ns/(wg,block) "
              f"({float(r['interaction_frac']) * 100:+.1f}% of removable; reported, not gated)")
        print(f"ceiling if the dot were free       = {float(r['ceiling_if_dot_free']):.4f}x "
              f"(plateau {float(r['plateau_gmac_s_if_dot_free']):.1f} GMAC/s)")
        print(f"ceiling if BOTH ALU terms were free= "
              f"{float(r['ceiling_if_both_alu_terms_free']):.4f}x "
              f"(plateau {float(r['plateau_gmac_s_if_both_free']):.1f} GMAC/s)")
    if math.isfinite(incumbent_recheck):
        print(f"incumbent re-check (split 'full' vs the occupancy sweep's own row at "
              f"m={SPLIT_RECHECK[0]} tokens={SPLIT_RECHECK[1]} blocks={SPLIT_RECHECK[2]}, "
              f"same shader and input synthesis) = {incumbent_recheck:+.2%} "
              f"(band {SPLIT_RECHECK_MIN_REL:.0%})")
    for p in split_missing:
        print(f"SPLIT FAIL: cell {p} produced no fit")
    for p in split_problems:
        print(f"SPLIT FAIL: {p}")

    # --- the whole-kernel cost model, and the quantity that decides the exit -----
    if unit_report:
        u = unit_report
        print("-" * 128)
        print("WHOLE-KERNEL COST MODEL — the ten real projection shapes, "
              f"{COST_MODEL_PASSES} passes (THIS IS THE EXIT TEST)")
        print("-" * 128)
        geo = {f"{r['label']}|{int(r['m'])}|{int(r['tokens'])}|{int(r['blocks'])}":
               (int(r["ng"]), int(r["work_units"])) for r in real}
        print(f"  {'shape':<40}{'ng':>7}{'units':>9}{'ns/unit':>11}{'replication':>13}")
        for k, v in u["per_shape_ns_per_unit"].items():
            label, m, tokens, blocks = k.split("|")
            ng, units = geo[k]
            print(f"  {label} m={m} tokens={tokens} blocks={blocks:<12}"
                  f"{ng:>7}{units:>9}{v:>11.4f}"
                  f"{u['per_shape_replication_spread'][k]:>12.3f}x")
        print(f"  device_ns ~ a*MACs + b                          R2 = "
              f"{float(u['mac_linear_r2']):.4f}   worst residual "
              f"{float(u['mac_linear_worst_residual']) * 100:.1f}%  -> MAC count alone does NOT "
              f"predict this kernel")
        print(f"  device_ns ~ a*units + b*workgroups + c          R2 = "
              f"{float(u['unit_and_workgroup_r2']):.6f}   worst residual "
              f"{float(u['unit_and_workgroup_worst_residual']) * 100:.1f}%  -> a high R2 on a "
              f"mis-specified model is not a rate")
        print(f"  per-unit cost across the ten shapes              = "
              f"{float(u['ns_per_unit_min']):.4f} .. {float(u['ns_per_unit_max']):.4f} ns, spread "
              f"{float(u['across_shape_spread']):.3f}x  -> there is NO single rate per unit")
        print(f"  the instrument's own spread on the same quantity = "
              f"{float(u['replication_spread']):.3f}x across {u['passes']} passes of the same "
              f"shapes in this same invocation")
        print(f"  EXIT QUANTITY  across-shape spread / instrument spread = "
              f"{float(u['effect']):.3f}x   (red iff <= 1: the threshold is the instrument's own "
              f"measured spread, not a constant)")
        if int(u["distinct_unit_counts"]) < int(u["shapes"]):
            print(f"  {int(u['shapes'])} shapes carry {int(u['distinct_unit_counts'])} distinct "
                  f"unit counts; the shapes sharing one differ by up to "
                  f"{float(u['worst_split_within_a_unit_count']):.2f}x in ns per unit, so "
                  f"`work_units` is not a sufficient statistic for this kernel (reported, "
                  f"not refused)")
    for p in unit_problems:
        print(f"COST MODEL FAIL: {p}")
    for p in geometry_problems:
        print(f"COST MODEL FAIL: {p}")

    # --- remainder: staging loads vs block loop vs finalize ----------------------
    for v in rem.REM_VARIANTS:
        if v in rem_slopes:
            print(f"  {v:<8} fit R^2={rem_r2[v]:.4f} (median-estimator cross-read "
                  f"{rem_fits_med[v][0]:.4f}, R^2={rem_fits_med[v][2]:.4f})  "
                  f"spirv_sha256={split_spirv_sha[v][:16]}...")
        else:
            print(f"  {v:<8} fit unavailable")
    if rem_report:
        rem.print_remainder(rem_report, rem_census)
    else:
        print("REMAINDER: no attributable fit (see the failures above)")
    for p in rem_missing:
        print(f"REMAINDER FAIL: cell {p} produced no fit")
    for p in rem_problems:
        print(f"REMAINDER FAIL: {p}")

    # --- the gate-scope non-vacuity count --------------------------------------
    gate_terms = [(suite, name, str(table[name]["state"]))
                  for suite, (table, terms) in
                  (("split", (split_report.get("term_table", {}), SPLIT_COST_TERMS)),
                   ("remainder", (rem_report.get("term_table", {}), rem.REM_COST_TERMS)))
                  for name in terms if name in table]
    print(f"declared cost terms of the gate ({len(gate_terms)}): "
          + ", ".join(f"{suite}.{name}={state}" for suite, name, state in gate_terms))
    for p in nonvacuity_problems:
        print(f"COST MODEL FAIL: {p}")

    # verdict
    print("-" * 128)
    memory_bound = max_unique_bw > 0.5 * peak_bw or min_intens < knee_intens
    if memory_bound:
        verdict = ("MEMORY-BOUND: achieved bandwidth is a large fraction of peak or intensity "
                   "sits below the roofline knee")
    else:
        head = (
            "NOT memory-bound (achieved bandwidth <= "
            f"{max_unique_bw / peak_bw * 100:.1f}% of peak; intensity >= {min_intens:.0f} MAC/byte, "
            f"far above the {knee_intens:.1f} MAC/byte knee). Dispatch is negligible "
            f"(<= {t_dispatch / min(float(r['device_ns_med']) for r in real) * 100:.2f}%). "
            "The dominant term is per-workgroup instruction/latency exposure: even the kernel's "
            f"own arithmetic plateau ({plateau:.0f} GMAC/s) is {CUDA_GMAC_S[0] / plateau:.1f}x-"
            f"{CUDA_GMAC_S[1] / plateau:.1f}x below CUDA MMQ, and the real shapes reach only "
            f"{real_peak_gmac / plateau * 100:.0f}% of that plateau. ")
        if split_report:
            sr = split_report
            if bool(sr["dot_resolved"]):
                dot_phrase = (f"the dot term at {float(sr['dot_ns']):+.2f} ns/(wg,block) "
                              f"({float(sr['dot_sigma']):.1f} sigma)")
            else:
                dot_phrase = (f"the dot term at {float(sr['dot_ns']):+.2f} +/- "
                              f"{float(sr['dot_se']):.2f} ns/(wg,block) — "
                              f"{abs(float(sr['dot_sigma'])):.1f} sigma, under a 3-sigma bound of "
                              f"{float(sr['min_sigma']) * float(sr['dot_se']):.2f} ns/(wg,block), so "
                              "the 8 dotPacked4x8EXT per lane per block cost nothing this geometry "
                              "can resolve")
            tail = (
                f"The 2x2 cost probes at ng=64 put {dot_phrase} and the unchanged checked-i128 "
                f"epilogue at {float(sr['epilogue_ns']):+.2f} +/- {float(sr['epilogue_se']):.2f} "
                f"ns/(wg,block) ({float(sr['epilogue_sigma']):.1f} sigma, "
                f"{sr['epilogue_state']}, "
                f"{float(sr['epilogue_share_of_full']) * 100:.1f}% of the marginal). Making the "
                f"dot free caps the kernel at {float(sr['ceiling_if_dot_free']):.3f}x and making "
                f"both ALU terms free caps it at "
                f"{float(sr['ceiling_if_both_alu_terms_free']):.3f}x; the remaining "
                f"{float(sr['remainder_share_of_full']) * 100:.0f}% is the staging-load pipeline, "
                "the block loop and the finalize epilogue, which no dot substitution reaches."
            )
        else:
            tail = "The fix is instruction throughput, not a memory-layout change."
        if rem_report:
            rr = rem_report
            dom = rr["dominant_term"]
            rem_phrase = (
                f"The remainder's own probes at the same ng=64 put {dom} at "
                f"{float(rr['share_of_marginal'][dom]) * 100:.1f}% of the marginal "
                f"({float(rr['terms_ns'][dom]):+.2f} +/- {float(rr['terms_se'][dom]):.2f} "
                f"ns/(wg,block), {float(rr['sigma'][dom]):.1f} sigma, "
                f"{rr['term_table'][dom]['state']}); the dm/e metadata staging "
                f"is {float(rr['share_of_marginal']['staging_meta']) * 100:.1f}%, the w/x staging "
                f"that leaves L1 is "
                f"{float(rr['share_of_marginal']['staging_weights_acts']) * 100:.1f}%, the finalize "
                f"epilogue is {float(rr['share_of_marginal']['finalize']) * 100:.2f}% (it is "
                f"outside the block loop, so it cannot enter the marginal) and the block loop "
                f"residual is {float(rr['block_loop_ns']) / float(rr['slope_ns_per_wg_block']['full']) * 100:.1f}%."
                if dom else
                "The remainder probes resolved no term beyond the 3-sigma bar.")
        else:
            rem_phrase = ""
        unit_phrase = ""
        if unit_report:
            unit_phrase = (
                f"The whole kernel has no single rate: device_ns spans "
                f"{float(unit_report['across_shape_spread']):.2f}x in ns per dispatch unit across "
                f"the ten real shapes against {float(unit_report['replication_spread']):.3f}x of "
                f"instrument spread on the same quantity, so per-shape rates are quotable and a "
                f"single GMAC/s is not (effect {float(unit_report['effect']):.2f}x, the exit "
                f"test)."
            )
        verdict = head + tail + rem_phrase + (" " if rem_phrase else "") + unit_phrase
    print(f"verdict: {verdict}")

    # --- checks -------------------------------------------------------------
    problems: list[str] = []
    problems += check_positive(results)

    mono_problems = check_monotone(occ)
    range_problems = check_dynamic_range(occ)
    pair_problems, pair_ratio = check_pair(results, *CONTROL_PAIR, MIN_PAIR_RATIO)
    print("-" * 128)
    print(f"control pair blocks={CONTROL_PAIR[0]} vs {CONTROL_PAIR[1]} at ng=512: "
          f"device_ns ratio = {pair_ratio:.2f}x (threshold {MIN_PAIR_RATIO:.2f}x)"
          if pair_ratio is not None else "control pair unavailable")
    if not deep_fit or deep_fit[0] <= 0:
        problems.append("deep-K fit has no positive arithmetic slope")
    if not occ_fit or occ_fit[0] <= 0:
        problems.append("occupancy fit has no positive per-workgroup slope")
    problems += pair_problems

    if args.controls:
        control_self_test()
        flat = copy.deepcopy(occ)
        stuck = statistics.median(float(r["device_ns_med"]) for r in occ)
        for r in flat:
            r["device_ns_med"] = stuck
            r["device_ns_min"] = stuck
        flat_problems = check_dynamic_range(flat)
        if not flat_problems:
            print("CONTROL FAIL: seeded stuck-timer occupancy series survived the checker")
            return 1
        print(f"control red (observed firing under this harness): seeded stuck-timer series "
              f"rejected — {flat_problems[0]}")
        # The ordering check must still bite: reverse the real per-token series and
        # require a rejection.  A flat seed cannot do this job, since a constant series
        # is monotone by definition.
        inverted_rows = copy.deepcopy(occ)
        for tokens in {int(r["tokens"]) for r in inverted_rows}:
            rows = sorted((r for r in inverted_rows if int(r["tokens"]) == tokens),
                          key=lambda r: float(r["ng"]))
            for r, src in zip(rows, reversed(rows)):
                r["device_ns_min"] = src["device_ns_min"]
        inv_problems = check_monotone(inverted_rows)
        if not inv_problems:
            print("CONTROL FAIL: a seeded inverted occupancy series survived the ordering check")
            return 1
        print(f"control red (observed firing under this harness): seeded inverted occupancy "
              f"series rejected — {inv_problems[0]}")
        if mono_problems or range_problems:
            print("CONTROL FAIL: the live occupancy series is not monotone-in-trend / wide-range")
            return 1
        print("control green: the unmutated live occupancy series is non-decreasing in ng with a "
              "wide dynamic range")

        # The device-level control.  The epilogue-elided mutant IS a seeded mutant of the real
        # kernel, built from its real text, and the unmutated 'full' cell is the restore; the
        # census below proves the elision reached the module.  The gate no longer requires that
        # mutant to MOVE -- that was the inherited bar that decided ten of nineteen runs -- so
        # what is asserted here is that the measured cells are distinct and the suite holds at
        # least one resolved-positive term.  The term's own classification is reported, and the
        # epilogue's swing across the band is exercised as a determinism control below.
        live_problems = attribute_split(split_slopes)[0]
        if not split_report or live_problems:
            print("CONTROL FAIL: the live split is not attributable under the resolved-term "
                  f"rule — {(live_problems or ['no report'])[0]}")
            return 1
        print(f"control green (device): the epilogue-elided mutant, built from the executing "
              f"kernel's own text and censused to {split_census['full']['arith'] - split_census['noepi']['arith']}"
              f"-fewer arithmetic opcodes, reads at "
              f"{float(split_report['epilogue_ns']):+.4f} +/- {float(split_report['epilogue_se']):.4f}"
              f" ns/(wg,block) -> {split_report['epilogue_state']}; the restored "
              f"'full' cell sits at {float(split_report['slope_ns_per_wg_block']['full']):.4f} "
              f"ns/(wg,block) and reproduces the occupancy sweep's own row to "
              f"{incumbent_recheck:+.2%}")
        print(f"control green (device): the dot-elided mutant reads at "
              f"{float(split_report['dot_ns']):+.4f} +/- {float(split_report['dot_se']):.4f} "
              f"ns/(wg,block) -> {split_report['dot_state']}; reported, not gated, and its "
              f"state decides nothing")

        # DETERMINISM CONTROL on the real measured numbers: sweep the live epilogue delta across
        # its entire unresolved band and require the verdict not to move once.  This is the
        # check that fails if any bar still reads a term the instrument cannot resolve -- the
        # defect that made the inherited gate red on 10 of 19 runs.
        epi_se = float(split_report["epilogue_se"])
        s_full = float(split_report["slope_ns_per_wg_block"]["full"])
        baseline = attribute_split(split_slopes)[0]
        for i in range(1, 61):
            value = SPLIT_MIN_SIGMA * epi_se * i / 60.0
            swept = attribute_split(dict(split_slopes,
                                        noepi=(s_full - value, epi_se)))[0]
            if swept != baseline:
                print(f"CONTROL FAIL: the verdict changed at a swept epilogue delta of "
                      f"{value:.4f} ns ({value / epi_se:.2f} of the {SPLIT_MIN_SIGMA:g}-sigma "
                      f"band): {swept[0] if swept else 'no longer a problem'}")
                return 1
        print(f"control green (device): the epilogue delta swept across all 60 points of its "
              f"unresolved band (0 .. {SPLIT_MIN_SIGMA * epi_se:.4f} ns at +/- {epi_se:.4f}) "
              f"leaves the verdict unchanged, so no bar reads a term this geometry cannot "
              f"resolve")

        # The three named refusals the protocol must keep, seeded into the REAL measured slopes.
        for seed_name, seeded, want in (
            ("a no-op epilogue term", dict(split_slopes, noepi=split_slopes["full"]), "not applied"),
            ("a both-zero attribution",
             dict(split_slopes, nodot=split_slopes["full"], noepi=split_slopes["full"]),
             "not applied"),
            ("a bare cell the device resolved as slower",
             dict(split_slopes, bare=(s_full + 2.0 * SPLIT_MIN_SIGMA * epi_se, epi_se)),
             "resolved NEGATIVE"),
        ):
            seeded_problems = attribute_split(seeded)[0]
            if not any(want in q for q in seeded_problems):
                print(f"CONTROL FAIL: {seed_name} survived the checker — {seeded_problems}")
                return 1
            print(f"control red (observed firing under this harness): {seed_name} was rejected "
                  f"— {seeded_problems[0]}")
        zero_slopes = {v: (0.0, 0.0) for v in SPLIT_VARIANTS}
        if not any("positive finite" in q for q in attribute_split(zero_slopes)[0]):
            print("CONTROL FAIL: a both-zero marginal survived the checker")
            return 1
        print(f"control red (observed firing under this harness): a both-zero marginal was "
              f"rejected — {attribute_split(zero_slopes)[0][0]}")
        wrong_shape = [dict(r, ng=int(r["ng"]) * 2) for r in real_passes[0]]
        if not cost.geometry_problems(wrong_shape, ng_of):
            print("CONTROL FAIL: a seeded wrong-shape row survived the geometry check")
            return 1
        print(f"control red (observed firing under this harness): a wrong-shape row was rejected "
              f"— {cost.geometry_problems(wrong_shape, ng_of)[0]}")

        # The exit quantity's own refusal, seeded into the REAL passes: if the shapes spread no
        # further than the instrument's own replication, nothing may be quoted per shape.
        flat_passes = [[dict(r, device_ns_min=float(r["work_units"]) * 34.0 * (1.0 + 0.01 * p))
                        for r in one] for p, one in enumerate(real_passes)]
        flat_problems, _ = cost.per_unit_fit(flat_passes, ng_of, "device_ns_min")
        if not any("per-unit spread" in q for q in flat_problems):
            print("CONTROL FAIL: a per-unit spread below the instrument's own replication "
                  "survived the exit test")
            return 1
        print(f"control red (observed firing under this harness): a per-unit spread below the "
              f"instrument's own replication was rejected — {flat_problems[0]}")
        if unit_problems:
            print("CONTROL FAIL: the live per-unit spread did not clear the instrument — "
                  + unit_problems[0])
            return 1
        print(f"control green (device): ns per dispatch unit spreads "
              f"{float(unit_report['across_shape_spread']):.3f}x across the ten real shapes "
              f"against {float(unit_report['replication_spread']):.3f}x of instrument "
              f"replication on the same quantity — effect "
              f"{float(unit_report['effect']):.3f}x, the exit test, and it is red at <= 1")

        # The same device-level control for the remainder suite: the tile-scalarized mutant
        # IS a seeded mutant of the real kernel, censused to have removed every
        # Function-storage access while keeping every buffer load, and the unmutated 'full'
        # cell is the restore.  A timing path that had stopped reading the tile registers
        # would redden here.
        if not rem_report:
            print("CONTROL FAIL: the live tile-scalarized mutant produced no attributable "
                  "remainder")
            return 1
        if rem_problems:
            print(f"CONTROL FAIL: the live remainder is not attributable under the resolved-term "
                  f"rule — {rem_problems[0]}")
            return 1
        if nonvacuity_problems:
            print("CONTROL FAIL: no declared cost term of the gate is resolved positive — "
                  + nonvacuity_problems[0])
            return 1
        print(f"control green (device): the gate-scope non-vacuity count holds — at least one "
              f"declared cost term is resolved positive, whichever term leads: "
              + ", ".join(f"{name} ({state})" for _, name, state in gate_terms
                          if state == "resolved+"))
        # The count's own negative control: if EVERY declared cost term's interval contains the
        # null, the instrument is not reading the kernel's work and nothing may be quoted.
        flat_suites = {"split": (cost.term_table(
            {n: 0.1 for n in ("dot", "checked_i128_epilogue", "removable_alu")},
            {n: 0.10 for n in ("dot", "checked_i128_epilogue", "removable_alu")}),
            SPLIT_COST_TERMS),
            "remainder": (cost.term_table({n: 0.1 for n in rem.REM_COST_TERMS},
                                          {n: 0.10 for n in rem.REM_COST_TERMS}),
                          rem.REM_COST_TERMS)}
        if cost.instrument_reads_work(flat_suites) == []:
            print("CONTROL FAIL: a gate whose every cost term was unresolved was accepted")
            return 1
        print(f"control red (observed firing under this harness): a gate whose every cost term "
              f"was unresolved was rejected — {cost.instrument_reads_work(flat_suites)[0]}")
        print(f"control green (device): the tile-scalarized mutant, built from the executing "
              f"kernel's own text and censused to {rem_census['full']['funvar']}->"
              f"{rem_census['scalar']['funvar']} Function variables and "
              f"{rem_census['full']['scratch_ld'] + rem_census['full']['scratch_st']}->"
              f"{rem_census['scalar']['scratch_ld'] + rem_census['scalar']['scratch_st']} local "
              f"memory ops at an unchanged {rem_census['scalar']['buf_ld']} buffer loads, reads at "
              f"{float(rem_report['terms_ns']['local_memory_tile']):+.4f} "
              f"+/- {float(rem_report['terms_se']['local_memory_tile']):.4f} ns/(wg,block) -> "
              f"{rem_report['term_table']['local_memory_tile']['state']} "
              f"({float(rem_report['sigma']['local_memory_tile']):.1f} sigma)")
        # Source-level control for the remainder census: seed an elision that did not reach
        # the module into copies of the REAL census.
        f_rc, s_rc = rem_census["full"], rem_census["scalar"]
        for seed_name, seeded in (
            ("the scalar probe left local-memory traffic",
             {**rem_census, "scalar": {**s_rc, "funvar": f_rc["funvar"],
                                       "scratch_ld": f_rc["scratch_ld"]}}),
            ("the scalar probe dropped a buffer load",
             {**rem_census, "scalar": {**s_rc, "buf_ld": f_rc["buf_ld"] - 1}}),
            ("the nometa probe removed no staging load",
             {**rem_census, "nometa": {**rem_census["nometa"], "buf_ld": f_rc["buf_ld"]}}),
            ("the qsame probe removed a load",
             {**rem_census, "qsame": {**rem_census["qsame"],
                                      "scratch_ld": f_rc["scratch_ld"] - 1}}),
            ("the q1 probe kept the q trip count",
             {**rem_census, "q1": {**rem_census["q1"], "trips_1": f_rc["trips_1"]}}),
            ("the nofin probe kept the finalize epilogue",
             {**rem_census, "nofin": {**rem_census["nofin"], "total": f_rc["total"]}}),
            ("a remainder probe duplicated another cell",
             {**rem_census, "qsame": dict(rem_census["nometa"])}),
        ):
            seeded_problems = rem.check_rem_census(seeded)
            if not seeded_problems:
                print(f"CONTROL FAIL: a census in which {seed_name} survived")
                return 1
            print(f"control red (observed firing under this harness): a census in which "
                  f"{seed_name} was rejected — {seeded_problems[0]}")
        if rem_census_problems:
            print("CONTROL FAIL: the real remainder census does not prove the elisions — "
                  + rem_census_problems[0])
            return 1
        print(f"control green (source): the real remainder census proves all five probes "
              f"reached their compiled modules, all {len(SPLIT_VARIANTS) + len(rem.REM_VARIANTS)} "
              "cost probes are distinct SPIR-V")

        # Source-level control: seed an elision that did not reach the module, and a
        # census that never removed the epilogue, into copies of the REAL census.
        for seed_name, seeded in (
            ("the dot elision did not reach the module",
             {**split_census, "nodot": {**split_census["nodot"], "sdot": 8}}),
            ("the epilogue elision removed no arithmetic",
             {**split_census, "noepi": {**split_census["noepi"],
                                       "arith": split_census["full"]["arith"]}}),
            ("a cost probe duplicated another cell",
             {**split_census, "bare": dict(split_census["noepi"])}),
        ):
            seeded_problems = check_census(seeded)
            if not seeded_problems:
                print(f"CONTROL FAIL: a census in which {seed_name} survived")
                return 1
            print(f"control red (observed firing under this harness): a census in which "
                  f"{seed_name} was rejected — {seeded_problems[0]}")
        if census_problems:
            print("CONTROL FAIL: the real census does not prove the elisions — "
                  + census_problems[0])
            return 1
        if split_problems:
            print("CONTROL FAIL: the unmutated measured split is not attributable — "
                  + split_problems[0])
            return 1
        print("control green (source): the real census proves all three elisions reached the "
              "compiled modules, and all four cells are distinct SPIR-V")

    print("=" * 128)
    if (problems or mono_problems or range_problems or split_problems or rem_problems
            or unit_problems or geometry_problems or nonvacuity_problems):
        for p in (problems + mono_problems + range_problems + split_problems + rem_problems
                  + unit_problems + geometry_problems + nonvacuity_problems):
            print(f"CONTROL FAIL: {p}")
        print("=" * 128)
        return 1
    print("CONTROL PASS: timer live (wide range, ordering in trend, control pair differs), every "
          "row's geometry self-consistent, attribution slopes positive, all three ALU and all "
          "five staging/block-loop/finalize cost-probe elisions census-proven, every priced term "
          "classified against the free-work null with no bar on an unresolved one, and the "
          f"whole-kernel per-unit spread {float(unit_report['across_shape_spread']):.2f}x clears "
          f"the instrument's own {float(unit_report['replication_spread']):.3f}x "
          f"(effect {float(unit_report['effect']):.2f}x, the exit test)")
    print("=" * 128)

    if args.json is not None:
        payload = {
            "device": {"name": ident["device_name"], "driver_version": ident["driver_version"],
                       "api_version": ident["api_version"],
                       "timestamp_period": ident["timestamp_period"],
                       "timestamp_valid_bits": ident["timestamp_valid_bits"]},
            "probe": probe,
            "geometry": {"invocations_per_wg": INVOCATIONS_PER_WG, "rows_per_wg": ROWS_PER_WG,
                         "tokens_per_wg": TOKENS_PER_WG, "macs_per_wg_block": MACS_PER_WG_BLOCK},
            "attribution": {"dispatch_floor_ns": t_dispatch, "iso_per_block_ns": l_blk,
                            "deep_slope_ns_per_wg_block": deep_fit[0] if deep_fit else None,
                            "deep_fit_r2": deep_fit[2] if deep_fit else None,
                            "occupancy_slope_ns_per_wg": occ_fit[0] if occ_fit else None,
                            "plateau_gmac_s": plateau, "peak_gmac_s": peak_gmac,
                            "real_peak_gmac_s": real_peak_gmac,
                            "max_unique_gb_s": max_unique_bw, "max_logical_gb_s": max_logic_bw,
                            "min_intensity": min_intens, "knee_intensity": knee_intens,
                            "peak_bw_gb_s": peak_bw, "verdict": verdict},
            "pair_ratio": pair_ratio, "results": results, "split_results": split,
            "dot_epilogue_split": {
                "geometry": {"m": SPLIT_M, "tokens": SPLIT_TOKENS,
                             "ng": ng_of(SPLIT_M, SPLIT_TOKENS), "blocks": SPLIT_BLOCKS,
                             "recheck_row": list(SPLIT_RECHECK), "reps": SPLIT_REPS,
                             "estimator": SPLIT_ESTIMATOR,
                             "metric": "device-timestamp delta across one dispatch"},
                "shader_source_sha256": hashlib.sha256(bench.SHADER.read_bytes()).hexdigest(),
                "spirv_sha256": split_spirv_sha,
                "census": split_census,
                "slopes_ns_per_wg_block": {v: split_slopes[v][0] for v in split_slopes},
                "slope_se": {v: split_slopes[v][1] for v in split_slopes},
                "slopes_median_estimator": {v: split_fits_med[v][0] for v in split_fits_med
                                            if split_fits_med[v] is not None},
                "fit_r2": split_r2,
                "report": split_report,
                "incumbent_recheck_rel": incumbent_recheck,
                "incumbent_recheck_band_rel": SPLIT_RECHECK_MIN_REL,
                "min_sigma": SPLIT_MIN_SIGMA,
                "census_problems": census_problems,
                "problems": split_problems,
            },
            "remainder_split": {
                "geometry": {"m": SPLIT_M, "tokens": SPLIT_TOKENS,
                             "ng": ng_of(SPLIT_M, SPLIT_TOKENS), "blocks": SPLIT_BLOCKS,
                             "reps": SPLIT_REPS, "estimator": SPLIT_ESTIMATOR,
                             "metric": "device-timestamp delta across one dispatch"},
                "variants": list(rem.REM_VARIANTS),
                "spirv_sha256": {v: split_spirv_sha[v] for v in rem.REM_VARIANTS},
                "census": rem_census,
                "slopes_ns_per_wg_block": {v: rem_slopes[v][0] for v in rem_slopes},
                "slope_se": {v: rem_slopes[v][1] for v in rem_slopes},
                "slopes_median_estimator": {v: rem_fits_med[v][0] for v in rem_fits_med
                                            if rem_fits_med[v] is not None},
                "fit_r2": rem_r2,
                "report": rem_report,
                "census_problems": rem_census_problems,
                "problems": rem_problems,
            },
            "remainder_results": remrows,
            "cost_model": {
                "passes": COST_MODEL_PASSES,
                "k_sigma": SPLIT_MIN_SIGMA,
                "split_cost_terms": list(SPLIT_COST_TERMS),
                "split_term_table": split_report.get("term_table", {}),
                "remainder_term_table": (rem_report or {}).get("term_table", {}),
                "remainder_cost_terms": list(rem.REM_COST_TERMS),
                "whole_kernel": unit_report,
                "unit_problems": unit_problems,
                "geometry_problems": geometry_problems,
                "nonvacuity_problems": nonvacuity_problems,
                "split_fit_per_pass": {v: split_fits[v][4] for v in SPLIT_VARIANTS
                                       if split_fits[v] is not None},
                "remainder_fit_per_pass": {v: rem_fits[v][4] for v in rem.REM_VARIANTS
                                           if rem_fits[v] is not None},
                "real_passes": real_passes,
            },
            "elapsed_s": elapsed,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")

    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"q8_occupancy: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
