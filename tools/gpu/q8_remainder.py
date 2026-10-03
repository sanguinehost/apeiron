#!/usr/bin/env python3
"""Cost probes for the deep-K remainder of the Apeiron Q8 SPIR-V kernel.

``tools/gpu/q8_occupancy.py`` splits the deep-K marginal of ``src/spirv/q8_tile.comp``
into the dot, the checked-i128 epilogue and a 96.6-97.8% remainder named by elimination
(staging loads, the block loop, the finalize epilogue).  This module measures that
remainder instead of naming it.

Five probes, each derived from the executing kernel's own text and each censused to prove
the elision reached the compiled module (``spirv-dis`` opcode census, run in-gate):

``scalar``  every tile register index forced to 0 (``wr``/``xt``/``dots``/``acc``/``flags``/
            ``mantissa``/``weightExp``/``actExp``).  The tiles are dynamically indexed
            local arrays, so the kernel keeps them in Function storage and every access is
            an OpAccessChain + OpLoad/OpStore -- local-memory traffic.  Constant indices
            let the same compiler promote them to SSA values, so this probe removes the
            local-memory traffic and nothing else: the buffer load count, the dot opcode
            and the loop trip counts are unchanged.  It is the ceiling of the route
            "scalarize the register tile into SSA values".
``nometa``  the four ``dm[]`` mantissa/exponent reads and the four ``e[]`` exponent reads
            replaced by constants.  Metadata staging.
``qsame``   the ``w[]``/``x[]`` addresses made independent of ``q``, so 48 distinct buffer
            loads collapse onto 8 addresses while every load instruction, every scratch
            access and every dot stays.  The delta is the cost of leaving L1, i.e. the
            memory-hierarchy component of the weight/activation staging.
``q1``      the ``q`` loop trip count 8 -> 1.  Removes 42 of the 48 ``w[]``/``x[]`` loads,
            7/8 of the dots and 7/8 of the ``q`` loop control in one edit: an upper bound
            on everything the ``q`` nest does beyond the dot.
``nofin``   the finalize epilogue's Q32.32 rounding and i64 range check replaced by a
            direct store of the accumulator.  It sits *outside* the block loop, so it can
            only appear in the fit intercept, never in the deep-K marginal.

Every probe keeps every buffer load live and the buffer access pattern of the weight and
activation staging identical to the executing kernel; they compute different values and
are cost probes, not kernels (see the trust boundary in the receipt).
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The interval width on a priced term, restated so this module owns one number for both suites.
# It is the width of the acceptance interval, not a size a term must reach: an unresolved term
# decides nothing (see tools/gpu/q8_costmodel.py).
REM_MIN_SIGMA = 3.0

# The terms a removal-of-work model predicts to be positive, and therefore the only ones whose
# classification can bite.  `finalize` is NOT among them: `q8_tile.comp:130-151` runs after the
# block loop closes, so it cannot enter a slope in ng*blocks and its probe is a perturbation of
# the loop rather than a subtraction from it.  `q1` and the elimination residual are upper bounds
# and a by-construction residual respectively; both are reported and decide nothing.
REM_COST_TERMS = ("local_memory_tile", "staging_meta", "staging_weights_acts")
REM_REPORTED_TERMS = ("finalize",)
# Static Function-storage loads+stores the executing kernel's loop bodies hold; the census
# requires the scalar probe to remove at least this many, or the probe is not the probe.
REM_CENSUS_MIN_SCRATCH = 8
REM_CENSUS_MIN_META_LOADS = 3      # the dm[]/e[] reads glslc emits statically
REM_CENSUS_MIN_TOTAL = 2           # qsame's address arithmetic
REM_CENSUS_MIN_FINALIZE = 20       # nofin's finalize arithmetic

REM_VARIANTS = ("scalar", "nometa", "qsame", "q1", "nofin")

# --- anchors: exact single-occurrence substrings of src/spirv/q8_tile.comp ---------
META_ANCHOR = (
    "            if (row < p.M) {\n"
    "                uint di = (b * p.M + row) * 2u;\n"
    "                mantissa[r] = int(dm[di]);\n"
    "                weightExp[r] = dm[di + 1u];\n"
    "            } else {\n")
META_PROBE = (
    "            if (row < p.M) {\n"
    "                mantissa[r] = 100;\n"
    "                weightExp[r] = 16u;\n"
    "            } else {\n")
EXP_ANCHOR = "            actExp[t] = token < p.tokens ? e[token * p.blocks + b] : ZERO_SCALE;\n"
EXP_PROBE = "            actExp[t] = token < p.tokens ? 16u : ZERO_SCALE;\n"
QLOOP_ANCHOR = "        for (uint q = 0u; q < 8u; ++q) {\n"
QLOOP_PROBE = "        for (uint q = 0u; q < 1u; ++q) {\n"
WADDR_ANCHOR = "                wr[r] = row < p.M ? int(w[(b * 8u + q) * p.M + row]) : 0;\n"
WADDR_PROBE = "                wr[r] = row < p.M ? int(w[(b * 8u) * p.M + row]) : 0;\n"
XADDR_ANCHOR = ("                xt[t] = token < p.tokens ? "
                "int(x[(token * p.blocks + b) * 8u + q]) : 0;\n")
XADDR_PROBE = ("                xt[t] = token < p.tokens ? "
               "int(x[(token * p.blocks + b) * 8u]) : 0;\n")
FINAL_ANCHOR = (
    "            if (flags[c] == 0u) {\n"
    "                // Q32.32 round-to-nearest with the source's positive 2^31 bias.\n"
    "                uvec4 rounded = acc[c];\n"
    "                uint64_t oldLow = uint64_t(rounded.x) | (uint64_t(rounded.y) << 32);\n"
    "                uint64_t low = oldLow + (uint64_t(1) << 31);\n"
    "                uint carry = uint(low < oldLow);\n"
    "                uint64_t high = (uint64_t(rounded.z) | (uint64_t(rounded.w) << 32)) + "
    "uint64_t(carry);\n"
    "                rounded = uvec4(uint(low), uint(low >> 32), uint(high), uint(high >> 32));\n"
    "                bool resultNeg = (rounded.z & 0x80000000u) != 0u;\n"
    "                bool fitsI64 = rounded.w == (resultNeg ? 0xffffffffu : 0u);\n"
    "                if (!fitsI64) flags[c] |= REFUSE_I64;\n"
    "                else out64[token * p.M + row] = uvec2(rounded.y, rounded.z);\n"
    "            }\n")
FINAL_PROBE = (
    "            if (flags[c] == 0u) {\n"
    "                out64[token * p.M + row] = uvec2(acc[c].y, acc[c].z);\n"
    "            }\n")

# The tile registers are dynamically indexed, so this probe cannot use a single-occurrence
# anchor: it uses an exact substitution count instead, checked against the source.
TILE_ARRAYS = ("mantissa", "weightExp", "actExp", "wr", "xt", "dots", "acc", "flags")
TILE_INDEX = re.compile(r"\b(" + "|".join(TILE_ARRAYS) + r")\[(?:r|t|c)\]")
TILE_INDEX_EXPECTED = 26


def build_rem_variants(text: str, source: Path) -> dict[str, str]:
    """Derive the five remainder cells from the executing kernel's real text."""
    for name, anchor in (("dm/e staging", META_ANCHOR), ("e[] staging", EXP_ANCHOR),
                         ("q loop", QLOOP_ANCHOR), ("w[] address", WADDR_ANCHOR),
                         ("x[] address", XADDR_ANCHOR), ("finalize", FINAL_ANCHOR)):
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError(
                f"q8_remainder: {source} holds {n} occurrences of the {name} anchor, not 1; "
                "a remainder cell would be an unmodified copy of the kernel and its cost term "
                "would read as free")
    n_idx = len(TILE_INDEX.findall(text))
    if n_idx != TILE_INDEX_EXPECTED:
        raise RuntimeError(
            f"q8_remainder: {source} holds {n_idx} dynamic tile-register indices, not "
            f"{TILE_INDEX_EXPECTED}; the scalar probe would scalarize {n_idx} of them and the "
            "scalar probe would no longer price the whole local-memory traffic. Re-derive "
            "TILE_INDEX_EXPECTED against the source.")
    scalar = TILE_INDEX.sub(r"\1[0]", text)
    return {
        "scalar": scalar,
        "nometa": text.replace(META_ANCHOR, META_PROBE).replace(EXP_ANCHOR, EXP_PROBE),
        "qsame": text.replace(WADDR_ANCHOR, WADDR_PROBE).replace(XADDR_ANCHOR, XADDR_PROBE),
        "q1": text.replace(QLOOP_ANCHOR, QLOOP_PROBE),
        "nofin": text.replace(FINAL_ANCHOR, FINAL_PROBE),
    }


def load_costmodel():
    """The acceptance protocol, loaded the way the gate loads this module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("q8_costmodel", HERE / "q8_costmodel.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load q8_costmodel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- static census: the non-vacuity witness for every remainder probe ---------------
def census_ext(spv: Path) -> dict[str, int]:
    """Local-memory and buffer-load census of one cell's SPIR-V.

    The 2x2 split census counts arithmetic opcodes, which is the right witness for an ALU
    elision.  These probes remove *memory* ops, so this census separates them: it resolves
    each OpLoad/OpStore pointer back to its storage root and counts Function-storage
    (local memory) traffic separately from Uniform/StorageBuffer traffic.
    """
    dis = spv.with_suffix(".dis")
    if not dis.exists():
        subprocess.run(["spirv-dis", "--no-color", str(spv), "-o", str(dis)],
                       check=True, capture_output=True, text=True)
    body = dis.read_text()
    fun = set(re.findall(r"(%\S+) = OpVariable \S+ Function\b", body))
    root: dict[str, str] = {}
    for m in re.finditer(r"(%\S+) = OpAccessChain \S+ (%\S+)", body):
        root[m.group(1)] = m.group(2)

    def storage(p: str) -> str:
        for _ in range(64):
            if p in fun or p not in root:
                return p
            p = root[p]
        return p

    scratch_ld = scratch_st = buf_ld = 0
    for line in body.splitlines():
        m = re.search(r"(%\S+) = OpLoad \S+ (%\S+)\s*$", line)
        if m:
            if storage(m.group(2)) in fun:
                scratch_ld += 1
            elif storage(m.group(2)) != m.group(2):
                buf_ld += 1
            continue
        m = re.search(r"OpStore (%\S+)", line)
        if m and storage(m.group(1)) in fun:
            scratch_st += 1
    return {
        "funvar": len(fun),
        "scratch_ld": scratch_ld,
        "scratch_st": scratch_st,
        "buf_ld": buf_ld,
        "chain": len(re.findall(r"= OpAccessChain\b", body)),
        "trips_1": len(re.findall(r"= OpULessThan %bool %\d+ %uint_1\b", body)),
        "trips_8": len(re.findall(r"= OpULessThan %bool %\d+ %uint_8\b", body)),
        "sdot": len(re.findall(r"= OpSDot\b", body)),
        "total": len(re.findall(r"^\s*(?:%\S+ = )?Op\w+", body, re.M)),
    }


def check_rem_census(census: dict[str, dict[str, int]]) -> list[str]:
    """Each remainder probe must have statically removed the work it prices.

    This is the only check that can fire on a source defect, and it is what makes a
    sub-sigma device delta mean "free" rather than "not applied".  The reference cell is
    ``full``: the executing kernel's own module.
    """
    if "full" not in census:
        return ["the remainder census has no 'full' reference cell"]
    f = census["full"]
    problems: list[str] = []
    for v in REM_VARIANTS:
        if v not in census:
            problems.append(f"remainder cell {v!r} is missing from the measured census")
    if problems:
        return problems
    c = census
    if f["funvar"] < 1:
        problems.append(f"the executing kernel holds {f['funvar']} Function-storage variables; "
                        "it has no dynamically indexed tile register to price")
    if f["scratch_ld"] + f["scratch_st"] < REM_CENSUS_MIN_SCRATCH:
        problems.append(f"the executing kernel holds {f['scratch_ld']} local loads and "
                        f"{f['scratch_st']} local stores, below the "
                        f"{REM_CENSUS_MIN_SCRATCH}-op bar; there is no local-memory traffic "
                        "to attribute")
    # scalar: every local-memory access gone, every buffer load and the dot untouched.
    s = c["scalar"]
    if s["funvar"] != 0 or s["scratch_ld"] or s["scratch_st"]:
        problems.append(f"the scalar cell still holds {s['funvar']} Function variables, "
                        f"{s['scratch_ld']} local loads and {s['scratch_st']} local stores; "
                        "the tile registers were not promoted to SSA values")
    if s["buf_ld"] != f["buf_ld"]:
        problems.append(f"the scalar cell changed the buffer load count: {s['buf_ld']} against "
                        f"the full cell's {f['buf_ld']}; the probe is not holding the staging "
                        "pattern fixed")
    if s["sdot"] != f["sdot"]:
        problems.append(f"the scalar cell holds {s['sdot']} dot opcodes against the full cell's "
                        f"{f['sdot']}; the probe changed the dot, not only the tile storage")
    # nometa: fewer buffer loads and fewer address chains, dot untouched.
    n = c["nometa"]
    if f["buf_ld"] - n["buf_ld"] < REM_CENSUS_MIN_META_LOADS:
        problems.append(f"the nometa cell removed {f['buf_ld'] - n['buf_ld']} buffer loads, "
                        f"below the {REM_CENSUS_MIN_META_LOADS}-load bar; the dm[]/e[] staging "
                        "reads are still in the module")
    if n["sdot"] != f["sdot"]:
        problems.append(f"the nometa cell holds {n['sdot']} dot opcodes against the full cell's "
                        f"{f['sdot']}")
    # qsame: fewer instructions, but every load and every scratch access retained.
    a = c["qsame"]
    if f["total"] - a["total"] < REM_CENSUS_MIN_TOTAL:
        problems.append(f"the qsame cell removed {f['total'] - a['total']} opcodes, below the "
                        f"{REM_CENSUS_MIN_TOTAL}-opcode bar; the address edit did not reach the "
                        "compiled module")
    if a["buf_ld"] != f["buf_ld"] or a["scratch_ld"] != f["scratch_ld"]:
        problems.append("the qsame cell changed a load count; the probe must keep every load "
                        "live and only move the addresses")
    # q1: one trip constant 8 -> 1.
    k = c["q1"]
    if k["trips_1"] != f["trips_1"] + 1 or k["trips_8"] != f["trips_8"] - 1:
        problems.append(f"the q1 cell's trip constants are {k['trips_1']}x1 and {k['trips_8']}x8 "
                        f"against the full cell's {f['trips_1']}x1 and {f['trips_8']}x8; the q "
                        "loop bound did not reach the compiled module")
    # nofin: the finalize arithmetic gone, everything in the block loop retained.
    z = c["nofin"]
    if f["total"] - z["total"] < REM_CENSUS_MIN_FINALIZE:
        problems.append(f"the nofin cell removed {f['total'] - z['total']} opcodes, below the "
                        f"{REM_CENSUS_MIN_FINALIZE}-opcode bar; the finalize epilogue is still "
                        "in the module")
    if z["sdot"] != f["sdot"] or z["scratch_ld"] != f["scratch_ld"]:
        problems.append("the nofin cell changed the dot or the local-memory traffic; the "
                        "finalize probe must touch nothing inside the block loop")
    return problems


# --- attribution -------------------------------------------------------------------
def _paired_slope(rows_a: list[dict[str, object]], rows_b: list[dict[str, object]],
                  field: str = "device_ns_min") -> tuple[float, float] | None:
    """Slope and standard error of the per-depth DIFFERENCE of two cells.

    The inherited 2x2 compares two independently fitted slopes, so its error is
    ``hypot(se_a, se_b)`` and every per-depth offset both cells share is counted twice.
    Both cells are measured at the same depths, so regressing the paired difference against
    the same ``work_units`` keeps the common offset out and is the lower-variance estimator
    of the same quantity.  Reported beside the inherited figure; never gated, because
    changing the inherited checker's estimator is a protocol change this task does not own.
    """
    a = {int(r["blocks"]): (float(r["work_units"]), float(r[field])) for r in rows_a}
    b = {int(r["blocks"]): (float(r["work_units"]), float(r[field])) for r in rows_b}
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return None
    xs = [a[k][0] for k in common]
    ys = [a[k][1] - b[k][1] for k in common]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    slope = (n * sxy - sx * sy) / denom
    icept = (sy - slope * sx) / n
    ss_res = sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys))
    sxx_c = sum((x - sx / n) ** 2 for x in xs)
    se = math.sqrt((ss_res / (n - 2)) / sxx_c) if (n > 2 and sxx_c > 0) else float("inf")
    return slope, se


def attribute_remainder(full: tuple[float, float],
                        slopes: dict[str, tuple[float, float]],
                        split_report: dict[str, object],
                        census: dict[str, dict[str, int]],
                        min_sigma: float = REM_MIN_SIGMA,
                        intercepts: dict[str, float] | None = None,
                        rows: dict[str, list[dict[str, object]]] | None = None
                        ) -> tuple[list[str], dict[str, object]]:
    """Decompose the deep-K remainder into named measured terms.

    ``full`` is the (slope, se) of the executing kernel's cell and ``slopes`` the same for
    each remainder cell; ``split_report`` carries the 2x2's dot and epilogue terms.  Each
    term is one probe's own delta, so each is an upper bound on the work that probe removes
    and nothing more.  The block loop is what is left over, named by elimination and
    reported with the interaction residual, because superposition is a hypothesis here.

    The only gated claim is non-vacuity: the ``scalar`` probe is the mutation this task is
    required to make, so its delta must clear ``min_sigma`` fit standard errors.  Which
    term dominates is reported, never asserted.
    """
    problems: list[str] = []
    report: dict[str, object] = {}
    for v in REM_VARIANTS:
        if v not in slopes:
            problems.append(f"remainder cell {v!r} is missing from the measured slopes")
    if problems:
        return problems, report
    s, e = {"full": float(full[0]), **{v: float(slopes[v][0]) for v in REM_VARIANTS}}, \
           {"full": float(full[1]), **{v: float(slopes[v][1]) for v in REM_VARIANTS}}
    for v in ("full",) + REM_VARIANTS:
        if not (math.isfinite(s[v]) and s[v] > 0.0):
            problems.append(f"remainder cell {v} marginal {s[v]:.4f} ns/(wg,block) is not a "
                            "positive finite slope; the remainder cannot be attributed")
    if problems:
        return problems, report

    dot = float(split_report["dot_ns"])
    epi = float(split_report["epilogue_ns"])
    terms: dict[str, float] = {}
    ses: dict[str, float] = {}
    for name, cell in (("local_memory_tile", "scalar"), ("staging_meta", "nometa"),
                       ("staging_weights_acts", "qsame"), ("finalize", "nofin")):
        terms[name] = s["full"] - s[cell]
        ses[name] = math.hypot(e["full"], e[cell])
    terms["dot"] = dot
    ses["dot"] = float(split_report["dot_se"])
    terms["checked_i128_epilogue"] = epi
    ses["checked_i128_epilogue"] = float(split_report["epilogue_se"])
    # q1 removes 7/8 of the q nest beyond the dot: an upper bound, reported not summed.
    q_bound = (8.0 / 7.0) * (s["full"] - s["q1"]) - dot
    q_se = (8.0 / 7.0) * math.hypot(e["full"], e["q1"])
    named = terms["local_memory_tile"] + terms["staging_meta"] + \
        terms["staging_weights_acts"] + terms["finalize"]
    block_loop = s["full"] - named - terms["dot"] - terms["checked_i128_epilogue"]

    # The resolved-term rule.  Every term is classified against the free-work null; only a term
    # the data RESOLVES can bite, and then only if it resolved negative.  There is no bar on a
    # term's size, so which term dominates is reported and never asserted.
    cost = load_costmodel()
    table = cost.term_table(terms, ses, min_sigma)
    states = {k: str(table[k]["state"]) for k in terms}
    resolved = {k: states[k] == cost.RESOLVED_POS for k in terms}
    ranked = sorted(((k, terms[k]) for k in terms if resolved[k]), key=lambda kv: -kv[1])
    dominant = ranked[0][0] if ranked else None

    problems += cost.distinct_cells_problems(
        {"full": (float(full[0]), float(full[1])),
         **{v: (float(slopes[v][0]), float(slopes[v][1])) for v in REM_VARIANTS}})
    problems += cost.resolved_term_problems(table, REM_COST_TERMS, "remainder")

    report = {
        "slope_ns_per_wg_block": s,
        "slope_se": e,
        "terms_ns": terms,
        "terms_se": ses,
        "resolved": resolved,
        "dominant_term": dominant,
        "share_of_marginal": {k: (terms[k] / s["full"]) for k in terms},
        "block_loop_ns": block_loop,
        "block_loop_se": math.sqrt(sum(ses[k] ** 2 for k in ses)),
        "block_loop_state": cost.classify(block_loop, math.sqrt(sum(ses[k] ** 2 for k in ses)),
                                          min_sigma),
        "term_table": table,
        "cost_terms": list(REM_COST_TERMS),
        "reported_terms": list(REM_REPORTED_TERMS),
        "q_nest_bound_ns": q_bound,
        "q_nest_bound_se": q_se,
        "remainder_ns": s["full"] - float(split_report["removable_ns"]),
        "interaction_ns": (named + terms["dot"] + terms["checked_i128_epilogue"]
                           - s["full"]),
        # The finalize epilogue sits OUTSIDE the block loop, so it can only enter the fit
        # intercept, never the slope.  Its intercept cross-read is in ns per dispatch, not
        # ns per (workgroup, block), and carries no standard error: linfit_se returns the
        # slope's error only.  Reported, never gated.
        "finalize_intercept_ns": ((intercepts["full"] - intercepts["nofin"])
                                  if intercepts else float("nan")),
        "min_sigma": min_sigma,
        "sigma": {k: (terms[k] / ses[k] if ses[k] > 0 else float("inf")) for k in terms},
        "census": census,
    }
    # Paired-difference cross-read: the same terms, estimated from the per-depth difference
    # of the two cells instead of from two independent fits.  Reported, never gated.
    paired: dict[str, tuple[float, float]] = {}
    if rows and "full" in rows:
        for name, cell in (("local_memory_tile", "scalar"), ("staging_meta", "nometa"),
                           ("staging_weights_acts", "qsame"), ("finalize", "nofin"),
                           ("checked_i128_epilogue", "noepi"), ("dot", "nodot")):
            got = _paired_slope(rows["full"], rows[cell]) if cell in rows else None
            if got is not None:
                paired[name] = got
    report["paired_ns"] = {k: v[0] for k, v in paired.items()}
    report["paired_se"] = {k: v[1] for k, v in paired.items()}
    report["paired_sigma"] = {k: (v[0] / v[1] if v[1] > 0 else float("inf"))
                              for k, v in paired.items()}
    return problems, report


def self_test() -> None:
    """Rejecting and accepting seeds for the remainder checker, run without a device."""
    full = (61.62, 0.10)
    # The shape this instrument is required to reproduce: the scalar probe moves the clock.
    good = {"scalar": (12.00, 0.10), "nometa": (58.00, 0.10), "qsame": (56.00, 0.10),
            "q1": (11.00, 0.10), "nofin": (61.50, 0.10)}
    split = {"dot_ns": 0.10, "dot_se": 0.15, "epilogue_ns": 1.60, "epilogue_se": 0.15,
             "removable_ns": 1.70}
    good_census = {"full": {"funvar": 8, "scratch_ld": 16, "scratch_st": 15, "buf_ld": 14,
                            "chain": 40, "trips_1": 0, "trips_8": 3, "sdot": 1, "total": 665},
                   "scalar": {"funvar": 0, "scratch_ld": 0, "scratch_st": 0, "buf_ld": 14,
                              "chain": 18, "trips_1": 0, "trips_8": 3, "sdot": 1, "total": 623},
                   "nometa": {"funvar": 8, "scratch_ld": 16, "scratch_st": 15, "buf_ld": 11,
                              "chain": 37, "trips_1": 0, "trips_8": 3, "sdot": 1, "total": 623},
                   "qsame": {"funvar": 8, "scratch_ld": 16, "scratch_st": 15, "buf_ld": 14,
                             "chain": 40, "trips_1": 0, "trips_8": 3, "sdot": 1, "total": 663},
                   "q1": {"funvar": 8, "scratch_ld": 16, "scratch_st": 15, "buf_ld": 14,
                          "chain": 40, "trips_1": 1, "trips_8": 2, "sdot": 1, "total": 665},
                   "nofin": {"funvar": 8, "scratch_ld": 16, "scratch_st": 15, "buf_ld": 14,
                             "chain": 41, "trips_1": 0, "trips_8": 3, "sdot": 1, "total": 630}}
    if check_rem_census(good_census):
        raise RuntimeError("remainder self-test: a census proving all five probes was rejected")
    probs, rep = attribute_remainder(full, good, split, good_census)
    if probs:
        raise RuntimeError(f"remainder self-test: a coherent remainder was rejected: {probs[0]}")
    if rep["dominant_term"] != "local_memory_tile":
        raise RuntimeError("remainder self-test: the seeded dominant term was not ranked first")
    # Anti-hypothesis controls: staging-dominant and block-loop-dominant seeds must ALSO be
    # accepted, so the checker cannot encode "the tile storage dominates".
    for name, seed in (
        ("staging-dominant", {"scalar": (60.00, 0.10), "nometa": (50.00, 0.10),
                              "qsame": (48.00, 0.10), "q1": (59.00, 0.10),
                              "nofin": (61.50, 0.10)}),
        # Every cell's marginal is distinct: two cells at the same marginal is the unapplied-probe
        # defect below, not a dominance seed.
        ("block-loop-dominant", {"scalar": (58.00, 0.10), "nometa": (60.10, 0.10),
                                 "qsame": (60.40, 0.10), "q1": (57.60, 0.10),
                                 "nofin": (61.50, 0.10)}),
    ):
        if attribute_remainder(full, seed, split, good_census)[0]:
            raise RuntimeError(f"remainder self-test: a {name} remainder was rejected; the "
                               "checker encodes a hypothesis about which term dominates")
    # The instrument-non-vacuity requirement, and it is a COUNT, not a term: a remainder in which
    # every cost term's interval contains the null says the clock is not reading the kernel.
    # A remainder with nothing resolved is not a defect on its own: the ALU suite can be
    # unresolved on this device in the low regime.  Non-vacuity is gate-scoped, so this is
    # refused by the GATE-level count, and only when nothing anywhere resolved.
    quiet = {"scalar": (61.50, 0.10), "nometa": (61.55, 0.10), "qsame": (61.58, 0.10),
             "q1": (61.52, 0.10), "nofin": (61.60, 0.10)}
    quiet_problems = attribute_remainder(full, quiet, split, good_census)
    if quiet_problems[0]:
        raise RuntimeError(f"remainder self-test: an all-unresolved remainder was rejected by the "
                           f"per-suite rule: {quiet_problems[0][0]}")
    cost = load_costmodel()
    if cost.instrument_reads_work({"remainder": (quiet_problems[1]["term_table"],
                                                 REM_COST_TERMS)}) == []:
        raise RuntimeError("remainder self-test: a gate whose remainder suite resolved nothing was "
                           "accepted")
    # Rejecting seeds, one per gate.
    noop_scalar = dict(good, scalar=full)
    if not attribute_remainder(full, noop_scalar, split, good_census)[0]:
        raise RuntimeError("remainder self-test: a no-op scalar probe survived")
    inverted = dict(good, scalar=(80.0, 0.10))
    if not any("resolved NEGATIVE" in x
               for x in attribute_remainder(full, inverted, split, good_census)[0]):
        raise RuntimeError("remainder self-test: a scalar probe the device resolved as SLOWER "
                           "than the full kernel survived")
    # The finalize probe sits outside the block loop, so a resolved-negative finalize term is a
    # reported perturbation and must not redden the gate.
    fin = dict(good, nofin=(66.00, 0.10))
    fin_problems = attribute_remainder(full, fin, split, good_census)[0]
    if any("finalize" in x for x in fin_problems):
        raise RuntimeError(f"remainder self-test: the finalize probe reddened the gate: {fin_problems[0]}")
    if not attribute_remainder(full, {k: v for k, v in good.items() if k != "qsame"},
                               split, good_census)[0]:
        raise RuntimeError("remainder self-test: a remainder missing a cell survived")
    if not attribute_remainder(full, {k: (0.0, 0.0) for k in good}, split, good_census)[0]:
        raise RuntimeError("remainder self-test: a zero-marginal remainder survived")
    for name, bad in (
        ("the scalar probe left local-memory traffic",
         {**good_census, "scalar": {**good_census["scalar"], "funvar": 8, "scratch_ld": 16}}),
        ("the scalar probe dropped a buffer load",
         {**good_census, "scalar": {**good_census["scalar"], "buf_ld": 11}}),
        ("the scalar probe changed the dot",
         {**good_census, "scalar": {**good_census["scalar"], "sdot": 0}}),
        ("the nometa probe removed no staging load",
         {**good_census, "nometa": {**good_census["nometa"], "buf_ld": 14}}),
        ("the qsame probe removed a load",
         {**good_census, "qsame": {**good_census["qsame"], "scratch_ld": 12}}),
        ("the q1 probe kept the q trip count",
         {**good_census, "q1": {**good_census["q1"], "trips_1": 0}}),
        ("the nofin probe kept the finalize epilogue",
         {**good_census, "nofin": {**good_census["nofin"], "total": 665}}),
        ("the executing kernel has no tile register to price",
         {**good_census, "full": {**good_census["full"], "scratch_ld": 2, "scratch_st": 2}}),
    ):
        if not check_rem_census(bad):
            raise RuntimeError(f"remainder self-test: a census where {name} survived")
    if not check_rem_census({k: v for k, v in good_census.items() if k != "nofin"}):
        raise RuntimeError("remainder self-test: a census missing a cell survived")
    print("remainder self-test PASS: five-cell census accepted, eight seeded elision defects "
          "and a missing cell rejected; coherent, staging-dominant and block-loop-dominant "
          "remainders accepted; a no-op cell (bit-identical to the kernel), a scalar probe "
          "resolved as slower, a zero-marginal and a missing-cell remainder rejected by name; "
          "an all-unresolved remainder accepted at suite scope and refused by the gate-scope "
          "non-vacuity count; and a resolved-negative finalize probe reported without reddening "
          "the gate", flush=True)


def print_remainder(report: dict[str, object], c: dict[str, dict[str, int]]) -> None:
    """Print the remainder table, its census and the ranked terms."""
    s = report["slope_ns_per_wg_block"]
    e = report["slope_se"]
    print("-" * 128)
    print("REMAINDER COST PROBES — staging / block loop / finalize, "
          "src/spirv/q8_tile.comp")
    print("-" * 128)
    for v in ("full",) + tuple(REM_VARIANTS):
        print(f"  {v:<8} marginal = {float(s[v]):>10.4f} ns/(wg,block)  +/- "
              f"{float(e[v]):.4f}  census fun={c[v]['funvar']} scratch_ld={c[v]['scratch_ld']} "
              f"scratch_st={c[v]['scratch_st']} buf_ld={c[v]['buf_ld']} total={c[v]['total']}")
    table = report["term_table"]
    cost = set(report["cost_terms"])
    print(f"  {'term':<24}{'ns/(wg,block)':>14}{'+/- se':>10}{'sigma':>9}"
          f"{'% of marginal':>15}  interval at k={float(report['min_sigma']):g} se"
          f"{'':>6}state")
    for k, v in sorted(report["terms_ns"].items(), key=lambda kv: -float(kv[1])):
        sh = float(report["share_of_marginal"][k]) * 100.0
        row = table[k]
        print(f"  {k:<24}{float(v):>14.4f}{float(report['terms_se'][k]):>10.4f}"
              f"{float(report['sigma'][k]):>9.2f}{sh:>14.2f}%  "
              f"[{float(row['lo']):+.4f}, {float(row['hi']):+.4f}]"
              f"{'  cost' if k in cost else '  rptd':>6}  {row['state']}")
    bl_state = report["block_loop_state"]
    print(f"  {'block loop (residual)':<24}{float(report['block_loop_ns']):>14.4f}"
          f"{float(report['block_loop_se']):>10.4f}"
          f"{float(report['block_loop_ns']) / float(report['block_loop_se']):>9.2f}"
          f"{float(report['block_loop_ns']) / float(s['full']) * 100:>14.2f}%  "
          f"{'':>26}  rptd   {bl_state}")
    print(f"  q-nest upper bound (q1, 8/7 scaled, dot removed) = "
          f"{float(report['q_nest_bound_ns']):.4f} +/- {float(report['q_nest_bound_se']):.4f} "
          f"ns/(wg,block); reported, not summed (it overlaps staging and tile storage)")
    print(f"  finalize epilogue, intercept cross-read      = "
          f"{float(report['finalize_intercept_ns']):.1f} ns per dispatch (full - nofin); it is "
          "outside the block loop, so it cannot enter the slope")
    print(f"  remainder after the 2x2 ALU terms (full - bare) = "
          f"{float(report['remainder_ns']):.4f} ns/(wg,block)")
    print(f"  superposition residual (sum of terms - full)   = "
          f"{float(report['interaction_ns']):+.4f} ns/(wg,block); reported, not gated")
    print(f"  paired-difference cross-read (same depths, common offset removed; reported, "
          f"never gated):")
    for k in sorted(report["paired_ns"], key=lambda k: -abs(float(report["paired_ns"][k]))):
        print(f"    {k:<24}{float(report['paired_ns'][k]):>9.4f} +/- "
              f"{float(report['paired_se'][k]):.4f}  {float(report['paired_sigma'][k]):>9.2f} "
              f"sigma")
    dom = report["dominant_term"]
    print(f"  dominant RESOLVED term = {dom} at "
          f"{float(report['share_of_marginal'][dom]) * 100:.1f}% of the marginal; reported, "
          f"never asserted -- the gate's exit does not depend on which term leads"
          if dom else "  dominant RESOLVED term = none; every interval contains the free-work null")