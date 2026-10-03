#!/usr/bin/env python3
"""Acceptance protocol for the Apeiron Q8 SPIR-V cost model.

``q8_occupancy.py`` prices the deep-K marginal of ``src/spirv/q8_tile.comp`` with cost probes,
and ``q8_remainder.py`` prices the remainder those probes left unnamed.  Both used to gate on
*inherited 2x2 bars*: a priced term had to clear three fit standard errors, and the two terms
that cleared it were the checked-i128 epilogue (0.25-18.0 sigma over 19 runs) and the ``bare``
delta built from it.  A bar that flips on a term the instrument cannot resolve decides an
acceptance on a number the data does not carry.

THIS MODULE replaces those bars with three statements, each derived from the measured
quantity and its measured uncertainty.  It holds no constant that was chosen to make a
measurement pass.

1. Measured uncertainty.  The deep-K sweep and the real-shape sweep are run ``PASSES`` times
   inside one invocation.  A term's uncertainty is the combination of the within-sweep fit
   error and the between-pass spread of the same slope: ``se = hypot(mean(se_pass), sd/sqrt(P))``.
   The between-pass term is what the inherited estimator omitted, and it is the component the
   two device regimes live in (the executing kernel's own marginal moved 4% between regimes
   while its within-sweep fit error read 0.11 ns).

2. The resolved-term rule.  Every priced term gets an interval ``[v - k*se, v + k*se]`` against
   the null "this probe's work is free", and exactly one of three states:
   ``unresolved`` (the interval contains the null), ``resolved+``, ``resolved-``.
   NO BAR FIRES ON AN ``unresolved`` TERM.  The only term-level requirement is coherence, and it
   is conditional: *if* a declared cost term is ``resolved-``, that is a contradiction -- the
   device resolved that removing work made the kernel slower -- and the gate is red.  An
   unresolved term is reported as free at this resolution and decides nothing.
   ``k = K_SIGMA`` is the width of the interval and nothing else.

3. The per-unit-spread exit.  The whole-kernel fit is accepted or refused by one number:
   the across-shape spread of ns per dispatch unit at the ten real projection shapes, divided
   by the instrument's own spread on the *same* quantity, measured by repeating one shape in
   the same invocation.  Red iff the ratio is <= 1 -- i.e. iff the shape dimension carries no
   effect the instrument can separate from its own noise.  The threshold is the instrument's
   own measured spread, not a hand-picked constant: it moves with the device, the clock and
   the estimator.

WHICH QUANTITY DECIDES EXIT, stated once: the term tables below never enter the exit test.
The exit test is (a) the per-unit-spread effect ratio and (b) the conditional coherence rule
plus one count -- at least one declared cost term must be ``resolved+``, which is the statement
that the clock is reading the kernel at all.  A term's magnitude and a term's sigma therefore
cannot flip the exit, which is the whole point: over 19 runs the epilogue term swung 0.25-18.0
sigma and decided the exit on ten of them.

WHAT IS NOT ASSERTED: which term dominates, that any term is positive, that a sum of the terms
decomposes the marginal (the probes are not independent -- their sum is 103-124% of it), and
that there is one rate per unit.  The last is measured false and reported: device_ns is not
linear in MACs for this kernel across the ten real shapes, so the gate prints the MAC-linear
R^2 next to the two-predictor R^2 and the per-unit spread next to both.
"""

from __future__ import annotations

import math

# Width of the acceptance interval on a priced term, in combined standard errors.  Three is the
# stated level, not a fitted quantity: the null is excluded at ~1e-3 one-sided under a Gaussian
# fit error.  No threshold in this module is derived from a term's measured size.
K_SIGMA = 3.0

# Repetitions of each sweep inside one invocation, and the minimum for a between-pass spread.
PASSES = 3

UNRESOLVED = "unresolved"
RESOLVED_POS = "resolved+"
RESOLVED_NEG = "resolved-"


# ----------------------------------------------------------------------- measured uncertainty


def replicated_fit(passes: list[list[dict[str, object]]], field: str = "device_ns_min",
                   xkey: str = "work_units"
                   ) -> tuple[float, float, float, float, list[dict[str, object]]] | None:
    """Slope from ``PASSES`` independent sweeps of the same depths, with a measured uncertainty.

    Each pass is fitted on its own, so the fit error and the pass-to-pass spread are separate
    quantities.  ``se_within`` is the mean of the passes' own slope standard errors and is NOT
    divided by sqrt(P): the passes share their depths, so their fit errors repeat rather than
    average away.  ``se_between`` is the standard error of the mean over passes.  The returned
    error is their quadrature sum, which is the interval a term's classification is read from.

    Returns ``(slope, mean_intercept, mean_r2, se, per_pass)`` or ``None`` if any pass cannot
    be fitted.
    """
    per_pass: list[dict[str, object]] = []
    for p, rows in enumerate(passes):
        ordered = sorted(rows, key=lambda r: int(r["blocks"]))
        fit = _linfit_se([float(r[xkey]) for r in ordered], [float(r[field]) for r in ordered])
        if fit is None:
            return None
        slope, icept, r2, se = fit
        per_pass.append({"pass": p, "slope": slope, "intercept": icept, "r2": r2, "se": se})
    slopes = [float(q["slope"]) for q in per_pass]
    n = len(slopes)
    mean = sum(slopes) / n
    se_within = sum(float(q["se"]) for q in per_pass) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in slopes) / (n - 1)
        se_between = math.sqrt(var / n)
    else:
        se_between = float("inf")
    return (mean, sum(float(q["intercept"]) for q in per_pass) / n,
            sum(float(q["r2"]) for q in per_pass) / n, math.hypot(se_within, se_between),
            per_pass)


def _linfit_se(xs: list[float], ys: list[float]) -> tuple[float, float, float, float] | None:
    """OLS ``(slope, intercept, R^2, slope standard error)``; ``None`` if under-determined."""
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
    ss_res = sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    sxx_c = sum((x - sx / n) ** 2 for x in xs)
    se = math.sqrt((ss_res / (n - 2)) / sxx_c) if (n > 2 and sxx_c > 0) else float("inf")
    return slope, icept, r2, se


def ols_r2(ys: list[float], *predictors: list[float]) -> tuple[float, list[float], float]:
    """``y ~ b0 + sum(b_i x_i)`` by normal equations: ``(R^2, beta, worst relative residual)``.

    Up to two predictors.  ``tools/gpu/incumbent_ratio_table.py`` carries a solver for the same
    two-predictor fit; it is not imported because that file belongs to another task and sets a
    module-level wall-clock deadline on import, which would put this gate's budget inside its
    540-second budget.  Duplication is two dozen lines of normal equations and buys an owned
    build path; the agreement of the two is checked by ``self_test`` below.
    """
    n, k = len(ys), len(predictors) + 1
    if n < k + 1:
        return (float("nan"), [], float("nan"))
    ata = [[0.0] * k for _ in range(k)]
    aty = [0.0] * k
    for i in range(n):
        row = [1.0] + [predictors[j][i] for j in range(len(predictors))]
        for r in range(k):
            aty[r] += row[r] * ys[i]
            for c in range(k):
                ata[r][c] += row[r] * row[c]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(ata[r][col]))
        if abs(ata[piv][col]) < 1e-300:
            return (float("nan"), [], float("nan"))
        ata[col], ata[piv] = ata[piv], ata[col]
        aty[col], aty[piv] = aty[piv], aty[col]
        for r in range(k):
            if r == col:
                continue
            f = ata[r][col] / ata[col][col]
            for c in range(col, k):
                ata[r][c] -= f * ata[col][c]
            aty[r] -= f * aty[col]
    beta = [aty[r] / ata[r][r] for r in range(k)]
    my = sum(ys) / n
    ss_res = ss_tot = 0.0
    worst = 0.0
    for i in range(n):
        pred = beta[0] + sum(beta[j + 1] * predictors[j][i] for j in range(len(predictors)))
        ss_res += (ys[i] - pred) ** 2
        ss_tot += (ys[i] - my) ** 2
        if ys[i] != 0.0:
            worst = max(worst, abs(ys[i] - pred) / abs(ys[i]))
    return (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), beta, worst)


# ------------------------------------------------------------------------- resolved-term rule


def classify(value: float, se: float, k: float = K_SIGMA) -> str:
    """One of :data:`RESOLVED_POS`, :data:`RESOLVED_NEG`, :data:`UNRESOLVED`.

    The null is "this probe's work is free", so a zero or non-finite uncertainty leaves the term
    unresolved rather than claiming a resolution no measurement supports: an interval of zero
    width would resolve every non-zero delta to infinity.
    """
    if not math.isfinite(se) or se <= 0.0:
        return UNRESOLVED
    if value > k * se:
        return RESOLVED_POS
    if value < -k * se:
        return RESOLVED_NEG
    return UNRESOLVED


def term_table(terms: dict[str, float], ses: dict[str, float],
               k: float = K_SIGMA) -> dict[str, dict[str, object]]:
    """Per-term ``(estimate, se, interval, state)`` against the free-work null."""
    out: dict[str, dict[str, object]] = {}
    for name in terms:
        v, se = float(terms[name]), float(ses[name])
        half = k * se if math.isfinite(se) else float("inf")
        out[name] = {"ns": v, "se": se, "lo": v - half, "hi": v + half,
                     "sigma": (v / se if se > 0 else float("inf")),
                     "state": classify(v, se, k)}
    return out


def distinct_cells_problems(slopes: dict[str, tuple[float, float]]) -> list[str]:
    """Two cells of one suite may not measure the same marginal bit for bit.

    The census proves an elision reached the compiled module; this is its timing twin.  Two
    distinct SPIR-V modules over the same depths cannot yield bit-identical fitted slopes, so a
    repeated value means the probe was not applied as its own module and every delta taken
    against it reads 0 -- the defect a seeded no-op term would carry, named.
    """
    seen: dict[float, str] = {}
    out: list[str] = []
    for name in sorted(slopes):
        value = float(slopes[name][0])
        if value in seen:
            out.append(
                f"the {name!r} cell's marginal is bit-identical to the {seen[value]!r} cell's "
                f"({value:.4f} ns/(wg,block)); two distinct cost probes cannot price identically, "
                f"so the {name!r} probe was not applied and its delta reads 0")
        else:
            seen[value] = name
    return out


def resolved_term_problems(table: dict[str, dict[str, object]],
                           cost_terms: tuple[str, ...] | list[str],
                           suite: str,
                           missing: list[str] | None = None) -> list[str]:
    """The only per-suite term requirement: conditional coherence.

    Coherence is conditional on the data resolving the term, so it cannot flip on a term the
    instrument cannot see.  ``cost_terms`` names the terms a removal-of-work model predicts to
    be positive; terms outside it (a probe outside the block loop, an elimination residual, an
    overlapping upper bound) are reported and decide nothing.

    The instrument-non-vacuity requirement is NOT here and is deliberately not per-suite: it is
    :func:`instrument_reads_work`, which ranges over every declared cost term of the gate.  A
    per-suite count flipped on this device -- in the low regime every one of the three ALU terms
    reads below its own interval while the remainder's terms resolve at 170+ sigma -- and a bar
    that flips on a 2.7%-of-marginal term is the defect this module was written to remove.
    """
    problems = [f"{suite} term {name!r} is missing from the measured term table" for name in missing
                or []]
    if problems:
        return problems
    for name in cost_terms:
        row = table[name]
        if row["state"] == RESOLVED_NEG:
            problems.append(
                f"the {name!r} term is resolved NEGATIVE at {float(row['sigma']):+.1f} sigma "
                f"({float(row['ns']):+.4f} +/- {float(row['se']):.4f} ns/(wg,block), interval "
                f"[{float(row['lo']):+.4f}, {float(row['hi']):+.4f}]): the device resolved that "
                f"removing this work made the kernel slower, which the cost model cannot hold")
    return problems


def instrument_reads_work(suites: dict[str, tuple[dict[str, dict[str, object]],
                                                 tuple[str, ...] | list[str]]]) -> list[str]:
    """At least one declared cost term of the WHOLE gate must be resolved positive.

    ``suites`` maps a suite name to its ``(term_table, cost_terms)``.  This is the one count in
    the protocol and it is a count, not a magnitude: it says the timing reads the kernel's work
    at all.  It is invariant to which term leads -- the staging-dominant, block-loop-dominant and
    ALU-dominant seeds are all accepted -- and it cannot flip on a term the data cannot resolve,
    because it fires only when EVERY declared term is below its own interval, which is a
    statement about the instrument rather than about any one term.
    """
    seen = 0
    for suite, (table, cost_terms) in sorted(suites.items()):
        for name in cost_terms:
            if name not in table:
                return [f"{suite} term {name!r} is missing from the measured term table"]
            seen += 1
            if table[name]["state"] == RESOLVED_POS:
                return []
    return [f"no declared cost term of the gate is resolved positive: all {seen} intervals "
            f"contain the free-work null at {K_SIGMA:g} sigma, so the timing is not reading the "
            f"kernel's work at this geometry"]


# ------------------------------------------------------------------- per-unit spread, whole kernel


def geometry_problems(rows: list[dict[str, object]], ng_of) -> list[str]:
    """A row must report the grid and unit count its own (M, tokens, blocks) imply.

    This is the whole-kernel fit's wrong-shape refusal: if a row's ``ng`` or ``work_units``
    disagrees with the geometry it claims, the per-unit costs below are not per-unit costs and
    the spread means nothing.
    """
    problems: list[str] = []
    for r in rows:
        m, tokens, blocks = int(r["m"]), int(r["tokens"]), int(r["blocks"])
        want_ng = int(ng_of(m, tokens))
        want_units = want_ng * blocks
        if int(r["ng"]) != want_ng:
            problems.append(
                f"row {r['label']!r} m={m} tokens={tokens} blocks={blocks} reports "
                f"ng={int(r['ng'])} against ceil(M/16)*ceil(tokens/32)={want_ng}; the row was "
                f"measured at a different shape than it reports")
        if int(r["work_units"]) != want_units:
            problems.append(
                f"row {r['label']!r} m={m} tokens={tokens} blocks={blocks} reports "
                f"work_units={int(r['work_units'])} against ng*blocks={want_units}; the unit "
                f"count the fit divides by is not this dispatch")
    return problems


def per_unit_fit(passes: list[list[dict[str, object]]], ng_of, field: str = "device_ns_min"
                 ) -> tuple[list[str], dict[str, object]]:
    """The whole-kernel cost model: ns per dispatch unit, and the instrument's own spread.

    The exit quantity.  ``across_spread`` is the spread of ns-per-unit over the shapes;
    ``replication_spread`` is the spread the instrument itself shows on ONE shape, from the
    passes measured in the same invocation.  ``effect = across / replication`` and the gate is
    red iff ``effect <= 1``: that is the only way the shape dimension can fail to be a
    measurable effect, and the threshold is the instrument's own measured spread rather than a
    constant chosen for the shapes on hand.

    The MAC-linear and two-predictor fits are reported beside it because device_ns is not linear
    in MACs for this kernel: a high R^2 on a mis-specified model is not evidence of a rate, so
    no R^2 floor is asserted anywhere.
    """
    problems: list[str] = []
    keys = [f"{r['label']}|{int(r['m'])}|{int(r['tokens'])}|{int(r['blocks'])}" for r in passes[0]]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        problems.append(f"per_unit_fit: shape {dupes[0]!r} appears {keys.count(dupes[0])} times in "
                        f"pass 0; the shape list carries a duplicate row and the spread is not "
                        f"across {len(keys)} shapes")
    if len(passes) < 2:
        problems.append(f"per_unit_fit: {len(passes)} passes, at least 2 are needed to measure "
                        f"the instrument's own spread")
        return problems, {}
    for p, rows in enumerate(passes[1:], start=1):
        got = [f"{r['label']}|{int(r['m'])}|{int(r['tokens'])}|{int(r['blocks'])}" for r in rows]
        if got != keys:
            problems.append(f"per_unit_fit: pass {p} carries {len(got)} rows against pass 0's "
                            f"{len(keys)}, and not the same shapes in the same order; the "
                            f"replication is not a replication")
    if problems:
        return problems, {}

    by_key: dict[str, list[float]] = {k: [] for k in keys}
    by_ns: dict[str, list[float]] = {k: [] for k in keys}
    for rows in passes:
        for k, r in zip(keys, rows):
            units = float(r["work_units"])
            if units <= 0:
                problems.append(f"per_unit_fit: row {k} carries {units:.0f} dispatch units; the "
                                f"per-unit cost is undefined")
                return problems, {}
            by_ns[k].append(float(r[field]))
            by_key[k].append(float(r[field]) / units)
    # Two shapes may carry the SAME dispatch-unit count and differ in cost -- the measured shape
    # set does exactly that (`ffn_gate_up' tokens=200 and `ffn_down' tokens=200 both dispatch
    # 50176 units and differ 2.8x in ns per unit), which is the finding that `work_units' is not
    # a sufficient statistic for this kernel.  It is reported, never refused.
    by_units: dict[float, list[str]] = {}
    for k, r in zip(keys, passes[0]):
        by_units.setdefault(float(r["work_units"]), []).append(k)
    shared = {u: v for u, v in by_units.items() if len(v) > 1}

    shape_value = {k: sum(v) / len(v) for k, v in by_key.items()}
    lo, hi = min(shape_value.values()), max(shape_value.values())
    across = (hi / lo) if lo > 0 else float("inf")
    repro = max(max(v) / min(v) for v in by_key.values() if min(v) > 0)
    effect = across / repro if repro > 0 else float("inf")
    worst_split = 0.0
    for group in shared.values():
        vals = [shape_value[k] for k in group]
        worst_split = max(worst_split, max(vals) / min(vals))

    # The two fits below are the R^2 cross-reads, on RAW device_ns against MAC count and against
    # (dispatch units, workgroup count).  They are reported, never floored: a high R^2 on a
    # mis-specified model is not evidence of a rate, and the two-predictor fit's own worst residual
    # is the reason.
    ns = [sum(by_ns[k]) / len(by_ns[k]) for k in keys]
    macs = [float(r["macs"]) for r in passes[0]]
    units = [float(r["work_units"]) for r in passes[0]]
    ngs = [float(r["ng"]) for r in passes[0]]
    r2_mac, _, worst_mac = ols_r2(ns, macs)
    r2_unit, beta, worst_unit = ols_r2(ns, units, ngs)
    report = {
        "field": field,
        "passes": len(passes),
        "shapes": len(keys),
        "ns_per_unit_min": lo, "ns_per_unit_max": hi, "across_shape_spread": across,
        "replication_spread": repro, "effect": effect,
        "mac_linear_r2": r2_mac, "mac_linear_worst_residual": worst_mac,
        "unit_and_workgroup_r2": r2_unit, "unit_and_workgroup_beta": beta,
        "unit_and_workgroup_worst_residual": worst_unit,
        "per_shape_device_ns": {k: sum(by_ns[k]) / len(by_ns[k]) for k in keys},
        "per_shape_ns_per_unit": shape_value,
        "per_shape_replication_spread": {k: max(v) / min(v) for k, v in by_key.items()},
        "distinct_unit_counts": len(by_units),
        "unit_counts_shared_by_shapes": {str(int(u)): v for u, v in shared.items()},
        "worst_split_within_a_unit_count": worst_split,
    }
    if not (math.isfinite(effect) and effect > 1.0):
        problems.append(
            f"per-unit spread: ns per dispatch unit spreads {across:.3f}x across the "
            f"{len(keys)} real shapes but the instrument reproduces one shape to only "
            f"{repro:.3f}x across the {len(passes)} passes in this same invocation "
            f"(effect {effect:.3f}x <= 1); the shape dimension is not a measurable effect and "
            f"no per-shape rate can be quoted from this run")
    return problems, report


# ------------------------------------------------------------------------------------ self-test


def self_test() -> None:
    """Seeds for the protocol itself: no device, no fixture data, every seed named.

    The determinism control is the load-bearing one: a term swept across the whole unresolved
    band must leave the problem list untouched, because a bar that flips inside the band is the
    defect this module was written to remove.
    """
    # --- the classifier: a resolved term and an unresolved term differ, deterministically ----
    if classify(36.0, 0.10) != RESOLVED_POS:
        raise RuntimeError("cost-model self-test: a 360-sigma term was not classified resolved+")
    if classify(0.09, 0.10) != UNRESOLVED:
        raise RuntimeError("cost-model self-test: a 0.9-sigma term was not classified unresolved")
    if classify(-4.0, 0.10) != RESOLVED_NEG:
        raise RuntimeError("cost-model self-test: a resolved negative term was not classified -")
    if classify(-0.30, 0.10) != UNRESOLVED:
        raise RuntimeError("cost-model self-test: a -3-sigma term was not classified unresolved")
    # The boundary: exactly k*se is unresolved, one ulp beyond it is resolved.  The comparison is
    # strict so the state is a step function, never a value-dependent shade.
    if classify(K_SIGMA * 0.10, 0.10) != UNRESOLVED:
        raise RuntimeError("cost-model self-test: a term exactly at k*se was called resolved")
    if classify(math.nextafter(K_SIGMA * 0.10, 1.0), 0.10) != RESOLVED_POS:
        raise RuntimeError("cost-model self-test: a term one ulp past k*se was not resolved")
    if classify(1.0, 0.0) != UNRESOLVED:
        raise RuntimeError("cost-model self-test: a zero-width interval claimed a resolution")

    # --- the term table and the conditional coherence rule -------------------------------
    terms = {"big": 36.0, "small": 0.9, "gone": -4.0}
    table = term_table(terms, {"big": 0.10, "small": 0.40, "gone": 0.10})
    if table["big"]["state"] != RESOLVED_POS or table["small"]["state"] != UNRESOLVED:
        raise RuntimeError("cost-model self-test: a resolved and an unresolved term were "
                           "classified the same way")
    if table["big"]["lo"] <= 0.0 <= table["big"]["hi"]:
        raise RuntimeError("cost-model self-test: a resolved term's interval contains the null")
    cost = ("big", "small")
    if resolved_term_problems(table, cost, "seed"):
        raise RuntimeError("cost-model self-test: a coherent, resolved-positive suite was "
                           "rejected by the coherence rule")
    neg = term_table(terms, {"big": 0.10, "small": 0.40, "gone": 0.10})
    if not resolved_term_problems(neg, ("big", "gone"), "seed"):
        raise RuntimeError("cost-model self-test: a resolved-negative cost term was accepted")
    # A term outside the declared cost set is reported and decides nothing, however it resolved.
    if resolved_term_problems(neg, ("big",), "seed"):
        raise RuntimeError("cost-model self-test: a term outside the cost set reddened the gate")
    # Non-vacuity is gate-scoped, so it accepts a suite in which nothing resolved IF a sibling
    # suite resolved something -- which is the low-regime shape this device actually produces.
    allfree = term_table({"a": 0.1, "b": 0.2}, {"a": 0.10, "b": 0.10})
    if resolved_term_problems(allfree, ("a", "b"), "seed"):
        raise RuntimeError("cost-model self-test: an all-unresolved suite was rejected by the "
                           "per-suite coherence rule; non-vacuity is gate-scoped")
    if instrument_reads_work({"seed": (allfree, ("a", "b"))}) == []:
        raise RuntimeError("cost-model self-test: a gate with nothing resolved was accepted")
    if instrument_reads_work({"alu": (allfree, ("a", "b")),
                              "staging": (table, ("big",))}):
        raise RuntimeError("cost-model self-test: a gate whose staging suite resolved something "
                           "was rejected on the ALU suite's behalf")
    if resolved_term_problems(table, ("big", "small"), "seed",
                              missing=["absent"])[0] != \
            "seed term 'absent' is missing from the measured term table":
        raise RuntimeError("cost-model self-test: a missing term was not named")

    # --- determinism: sweep a term across its whole unresolved band ------------------------
    # The defect being replaced was a bar that flipped on a term the data cannot resolve.  Here
    # the unresolved term is swept from 0 to k*se in 60 steps; the problem list must be the same
    # empty list at every one of them, and the resolved term must stay resolved throughout.
    ref = None
    for i in range(61):
        value = K_SIGMA * 0.10 * i / 60.0
        row = term_table({"big": 36.0, "small": value}, {"big": 0.10, "small": 0.10})
        if row["small"]["state"] != UNRESOLVED:
            raise RuntimeError(f"cost-model self-test: {value:.4f} ns at 1.0 sigma was classified "
                               f"{row['small']['state']}; the band edge is not the boundary")
        got = resolved_term_problems(row, ("big", "small"), "seed")
        if ref is None:
            ref = got
        elif got != ref:
            raise RuntimeError("cost-model self-test: the problem list changed inside the "
                               "unresolved band; a bar still flips on an unresolvable term")
    if ref:
        raise RuntimeError("cost-model self-test: the swept band was rejected somewhere")

    # --- bit-identical cells: the seeded no-op term, named ----------------------------------
    if not distinct_cells_problems({"full": (61.8576, 0.10), "noepi": (61.8576, 0.10)}):
        raise RuntimeError("cost-model self-test: a no-op probe survived (bit-identical cells)")
    if not any("not applied" in p
               for p in distinct_cells_problems({"full": (61.8576, 0.10),
                                                 "nodot": (61.4, 0.10),
                                                 "bare": (61.8576, 0.11)})):
        raise RuntimeError("cost-model self-test: a repeated marginal was not named")
    if distinct_cells_problems({"full": (61.8576, 0.10), "nodot": (61.4, 0.10),
                                "noepi": (59.7, 0.10), "bare": (60.1, 0.10)}):
        raise RuntimeError("cost-model self-test: four distinct cells were reported as repeated")

    # --- measured uncertainty: the between-pass component is what the inherited se omitted ---
    deep = [{"blocks": 64, "work_units": 4096, "device_ns_min": 4096 * 60.0},
            {"blocks": 160, "work_units": 10240, "device_ns_min": 10240 * 60.0},
            {"blocks": 320, "work_units": 20480, "device_ns_min": 20480 * 60.0}]
    steady = replicated_fit([deep, deep, deep])
    if steady is None:
        raise RuntimeError("cost-model self-test: three identical passes could not be fitted")
    if not (math.isclose(steady[0], 60.0, rel_tol=1e-9) and math.isclose(steady[3], 0.0, abs_tol=1e-9)):
        raise RuntimeError(f"cost-model self-test: identical passes gave se={steady[3]}, not 0")
    # The same three depths with one cell's clock 4% higher in one pass: the regime shift the
    # inherited estimator's 0.11 ns could not see.
    shifted = [list(deep), [dict(r, device_ns_min=r["device_ns_min"] * 1.04) for r in deep], deep]
    got = replicated_fit(shifted)
    if got is None or not (0.3 < got[3] < 0.9):
        raise RuntimeError(f"cost-model self-test: a 4% between-pass shift read se={got[3] if got else None}"
                           ", expected the 0.27 ns the shift implies; the measured uncertainty "
                           "does not carry the device regime")
    if replicated_fit([deep]) is None or not math.isinf(replicated_fit([deep])[3]):
        raise RuntimeError("cost-model self-test: a single pass claimed a finite uncertainty")

    # --- the per-unit spread exit ----------------------------------------------------------
    def ng_of(m: int, tokens: int) -> int:
        return -(-m // 16) * -(-tokens // 32)

    def row(label, m, tokens, blocks):
        ng = ng_of(m, tokens)
        return {"label": label, "m": m, "tokens": tokens, "blocks": blocks, "ng": ng,
                "work_units": ng * blocks, "macs": float(m * tokens * blocks * 32)}

    shapes = [("attn_q", 4096, 200, 32), ("attn_q", 4096, 512, 32),
              ("attn_k_v", 512, 200, 32), ("attn_k_v", 512, 512, 32),
              ("attn_output", 1024, 200, 64), ("attn_output", 1024, 512, 64),
              ("ffn_gate_up", 3584, 200, 32), ("ffn_gate_up", 3584, 512, 32),
              ("ffn_down", 1024, 200, 112), ("ffn_down", 1024, 512, 112)]
    # ns per dispatch unit spanning 3x across shapes, with a 1% instrument spread within a shape.
    rates = [34.0, 34.0, 26.0, 26.0, 60.0, 60.0, 28.0, 28.0, 70.0, 70.0]
    live = []
    for scale in (1.0, 1.01, 0.99):
        live.append([dict(row(l, m, t, b),
                          device_ns_min=row(l, m, t, b)["work_units"] * rates[i] * scale)
                     for i, (l, m, t, b) in enumerate(shapes)])
    problems, report = per_unit_fit(live, ng_of)
    if problems:
        raise RuntimeError(f"cost-model self-test: a shape effect of 3x was rejected: {problems[0]}")
    if not (2.0 < float(report["across_shape_spread"]) < 3.5):
        raise RuntimeError(f"cost-model self-test: across-shape spread {report['across_shape_spread']}")
    if not (1.0 < float(report["replication_spread"]) < 1.03):
        raise RuntimeError(f"cost-model self-test: replication spread {report['replication_spread']}")
    # The seed the control exists for: the shapes differ but no more than the instrument's own
    # replication, so nothing may be quoted per shape.  This is the wrong-shape/flat-world case.
    flat = [[dict(r, device_ns_min=float(r["work_units"]) * 34.0 * (1.0 + 0.01 * p)) for r in one]
            for p, one in enumerate(live)]
    fprob, frep = per_unit_fit(flat, ng_of)
    if not fprob or not any("per-unit spread" in x for x in fprob):
        raise RuntimeError("cost-model self-test: a per-unit spread below the instrument's own "
                           "replication survived")
    if float(frep["effect"]) > 1.0:
        raise RuntimeError("cost-model self-test: the seeded flat-world run did not land at "
                           "effect <= 1")
    if not geometry_problems([dict(row("x", 128, 256, 64), ng=512)], ng_of):
        raise RuntimeError("cost-model self-test: a row whose ng disagrees with its M and tokens "
                           "survived")
    if not geometry_problems([dict(row("x", 128, 256, 64), work_units=999)], ng_of):
        raise RuntimeError("cost-model self-test: a row whose work_units disagrees with ng*blocks "
                           "survived")
    if geometry_problems(live[0], ng_of):
        raise RuntimeError("cost-model self-test: consistent rows were reported as wrong-shape")
    dprob, _ = per_unit_fit([live[0], live[0][:5]], ng_of)
    if not any("same shapes" in x for x in dprob):
        raise RuntimeError("cost-model self-test: a pass carrying different shapes survived")
    dup = [live[0][0], live[0][1]] + live[0][1:]
    uprob, _ = per_unit_fit([dup, dup], ng_of)
    if not any("duplicate row" in x for x in uprob):
        raise RuntimeError("cost-model self-test: a duplicated shape row survived")

    # --- the two-predictor fit agrees with a single-predictor fit on one predictor ----------
    ys, xs = [13.0, 21.0, 29.0, 37.0, 45.0, 53.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    r2a = ols_r2(ys, xs)[0]
    if abs(r2a - 1.0) > 1e-9:
        raise RuntimeError(f"cost-model self-test: the exact fit gave R2={r2a}, not 1")
    # A collinear second predictor is a rank-deficient system and must REFUSE, not return a
    # number: `abs(nan - 1) > 1e-9` is False, so a nan would slip through a `> 1e-9` test.
    if not math.isnan(ols_r2(ys, xs, [2.0 * v for v in xs])[0]):
        raise RuntimeError("cost-model self-test: a collinear two-predictor fit returned an R2")
    # An independent second predictor on an exact two-term series is still exact.
    if abs(ols_r2(ys, xs, [v * v for v in xs])[0] - 1.0) > 1e-9:
        raise RuntimeError("cost-model self-test: the exact two-predictor fit did not give R2=1")
    # A cubic in a single predictor is a legitimate fit and the solver must return it; it is not
    # MAC-linear (R2 0.88, worst residual 32%), which is why no R2 floor is asserted anywhere.
    if not (0.85 < ols_r2([v ** 3 for v in xs], xs)[0] < 0.90):
        raise RuntimeError("cost-model self-test: the single-predictor cubic fit moved; the "
                           "reference figure the receipt quotes has changed")
    if not (31.0 < ols_r2([v ** 3 for v in xs], xs)[2] < 33.0):
        raise RuntimeError("cost-model self-test: the cubic fit's worst residual moved")
    if not math.isnan(ols_r2(ys, [1.0] * 6)[0]):
        raise RuntimeError("cost-model self-test: a constant predictor returned an R2")
    if not math.isnan(ols_r2([1.0, 2.0], xs)[0]):
        raise RuntimeError("cost-model self-test: an under-determined fit returned an R2")

    print("cost-model self-test PASS: non-vacuity is gate-scoped (an all-unresolved suite is "
          "accepted when a sibling suite resolved); classifier separates a 360-sigma term from a 0.9-sigma one "
          "and is strict at the 3-sigma edge; the problem list is invariant across all 61 points "
          "of an unresolved term's band; resolved-negative is the only term-level red, and only "
          "for a declared cost term; a bit-identical cell pair is named as an unapplied probe; a "
          "4% between-pass shift reads as a finite uncertainty where three identical passes read "
          "zero; a 3x across-shape per-unit spread clears a 1% instrument replication, a "
          "flat-world run at effect<=1 is refused, and two wrong-shape seeds are named; the "
          "two-predictor OLS is exact on one predictor and degenerate otherwise", flush=True)