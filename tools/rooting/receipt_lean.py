#!/usr/bin/env python3
"""Generate proof/corpus/Perf/Receipts/Onebit.lean from the committed one-bit receipts.

Input: the receipts in RECEIPTS below, each bound to its ledger row in docs/evidence/ledger.jsonl
(id `onebit-receipt-<file stem>`). Numbers are read as JSON decimal text (parse_float=str) and
converted to exact rationals with Fraction. Means are defined in Lean, and the Python mean is
emitted only as the right-hand side of a norm_num theorem.

Modes:
  --check       regenerate in memory; exit 1 on a stale receipt sha256 (ledger) or on drift
                between the generated text and the committed file (first differing line)
  --write       write the generated file
  --self-test   run the controls on temporary copies; exit 0 only if the baseline is green and
                each control is red
  --root DIR    operate on another tree with the same layout (used by --self-test)
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = "docs/evidence/ledger.jsonl"
OUT = "proof/corpus/Perf/Receipts/Onebit.lean"

T3_ARM = "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp-nb"
RESQ_ARM = "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-resq-qmlp-nb"
T0B_ARM = "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm"
T0_ARM = "had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp"

# Lean namespace id -> (receipt path, variants arm key or None for the side-information receipt)
RECEIPTS = {
    "t3_nb_3seed": ("docs/evidence/receipts/onebit/t3-nb-3seed.json", T3_ARM),
    "w18_rowsel_resid_3seed": ("docs/evidence/receipts/onebit/w18-rowsel-resid-3seed.json", T3_ARM),
    "w18_resq_3seed": ("docs/evidence/receipts/onebit/w18-resq-3seed.json", RESQ_ARM),
    "t0b_sinkfnorm_3seed": ("docs/evidence/receipts/onebit/t0b-sinkfnorm-3seed.json", T0B_ARM),
    "t0_best_3seed": ("docs/evidence/receipts/onebit/t0-best-3seed.json", T0_ARM),
    "t6_sideinfo": ("docs/evidence/receipts/onebit/t6-sideinfo.json", None),
}

# W17-C receipts: namespace id -> receipt path. Emitted after RECEIPTS by w17_fields().
W17 = {
    "w17_c_fit": "docs/evidence/receipts/onebit/w17-c-fit.json",
    "w17_c_measure100": "docs/evidence/receipts/onebit/w17-c-measure100.json",
    "w17_c_run25": "docs/evidence/receipts/onebit/w17-c-run25.json",
    "w17_c_run50": "docs/evidence/receipts/onebit/w17-c-run50.json",
    "w17_c_run100": "docs/evidence/receipts/onebit/w17-c-run100.json",
    "w17_d_tilt": "docs/evidence/receipts/onebit/w17-d-tilt.json",
}

# W15 receipts for the §12 and §28 verdicts (the backfit arm is single-seed; t3-base-seeds12 holds seeds 1, 2 only).
W15 = {
    "w15_backfit_attn": "docs/evidence/receipts/onebit/2026-10-12-fold-transfer-attn-bscale.json",
    "t3_base_seeds12": "docs/evidence/receipts/onebit/t3-base-seeds12.json",
    "t6_sideinfo_ext": "docs/evidence/receipts/onebit/t6-sideinfo.json",
    "w18_outcomes": "docs/evidence/receipts/onebit/w18-outcomes.json",
    "w18_kd_main1": "docs/evidence/receipts/onebit/w18-kd-main1.json",
    "w18_kd_main2": "docs/evidence/receipts/onebit/w18-kd-main2.json",
    "w18_calx600": "docs/evidence/receipts/onebit/w18-calx600.json",
    "w18_trellis2": "docs/evidence/receipts/onebit/w18-trellis2.json",
    "w18_size05": "docs/evidence/receipts/onebit/w18-size05-n50.json",
    "w18_size15": "docs/evidence/receipts/onebit/w18-size15-n50.json",
    "w18_size15m": "docs/evidence/receipts/onebit/w18-size15-m792.json",
    "w18_pv05": "docs/evidence/receipts/onebit/w18-pv05.json",
    "w18_pvobs05": "docs/evidence/receipts/onebit/w18-pvobs05.json",
    "w18_pvlam05": "docs/evidence/receipts/onebit/w18-pvlam05.json",
    "w18_pvste05": "docs/evidence/receipts/onebit/w18-pvste05.json",
    "w18_pvpre05": "docs/evidence/receipts/onebit/w18-pvpre05.json",
    "w18_pvabnorm05": "docs/evidence/receipts/onebit/w18-pvabnorm05.json",
    "w18_pvabr05": "docs/evidence/receipts/onebit/w18-pvabr05.json",
    "w18_pvaball05v0": "docs/evidence/receipts/onebit/w18-pvaball05v0.json",
    "w18_pvabrcd05": "docs/evidence/receipts/onebit/w18-pvabrcd05.json",
    "w18_pvabcd05": "docs/evidence/receipts/onebit/w18-pvabcd05.json",
    "w18_pvgrid25": "docs/evidence/receipts/onebit/w18-pvgrid25.json",
    "w18_pvtok146": "docs/evidence/receipts/onebit/w18-pvtok146.json",
    "w18_pvtok292": "docs/evidence/receipts/onebit/w18-pvtok292.json",
    "w18_pvfast": "docs/evidence/receipts/onebit/w18-pvfast.json",
    "w18_pvprof": "docs/evidence/receipts/onebit/w18-pvprof.json",
    "w18_pvprof2": "docs/evidence/receipts/onebit/w18-pvprof2.json",
    "w18_hadmm": "docs/evidence/receipts/onebit/w18-hadmm.json",
    "w18_guided05": "docs/evidence/receipts/onebit/w18-guided05.json",
    "w18_guidedsweep": "docs/evidence/receipts/onebit/w18-guidedsweep.json",
    "w18_guidedseed": "docs/evidence/receipts/onebit/w18-guidedseed.json",
    "qreg_kan_probe": "docs/evidence/receipts/onebit/qreg-kan-probe.json",
}

# End-loss-weighted Gram receipts (fold_transfer "-guided[a|b]NN"): namespace -> variant prefixes (see guided_prefix).
GUIDED = {"w18_guided05": ["base", "gab100", "gab50"], "w18_guidedsweep": ["gab25", "gab75", "ga50", "gb100"],
          "w18_guidedseed": ["base", "gab50"]}
# Multi-seed guided receipts (calibration-order seeds): namespace -> seeds, emitted per seed as `<prefix><Field>S<seed>`.
GUIDED_SEEDS = {"w18_guidedseed": ["1", "2"]}

# Scale-group ablation of the finishing drop (pv_finish.py --train, V-steps off): namespace -> trained groups.
PVAB_GROUPS = {"w18_pvabnorm05": ["norm"], "w18_pvabr05": ["r"], "w18_pvaball05v0": ["c", "d", "norm", "r"], "w18_pvabrcd05": ["c", "d", "r"], "w18_pvabcd05": ["c", "d"],
               "w18_pvgrid25": ["c", "d", "norm", "r"], "w18_pvtok146": ["c", "d", "norm", "r"], "w18_pvtok292": ["c", "d", "norm", "r"],
               "w18_pvfast": ["c", "d", "norm", "r"], "w18_hadmm": ["c", "d", "norm", "r"]}

# Ten-step torch.profiler windows of the finishing loop (pv_finish.py --profile 10): namespace -> (compose, kl) path.
# `selfDeviceS` is the receipt's own total: pvprof sums aten-op and kernel self time (double counts), pvprof2 kernels only.
# The aten-level sums below are the same method in both: self device time of the receipt's listed aten ops.
PROFILES = {"w18_pvprof": ("layer", "ckpt"), "w18_pvprof2": ("step", "fused")}
PROFILE_ELEMENTWISE = ["aten::mul", "aten::cat", "aten::sub", "aten::add", "aten::add_", "aten::copy_", "aten::fill_", "aten::index",
                       "aten::gather"]

OUTCOME_ARMS = [("teacher", "teacher"), ("q4km", "q4km"), ("resq", RESQ_ARM)]
OUTCOME_RAT = [("C4Ppl", "c4_ppl"), ("Wt2Ppl", "wt2_ppl"), ("ArcAcc", "arc_acc"), ("ArcAccNorm", "arc_acc_norm"),
               ("HsAcc", "hs_acc"), ("HsAccNorm", "hs_acc_norm")]

# 2026-10-11 escalation receipts (copied with their names): the W15a/W15b rate-distortion, ray and K-FAC
# verdicts in CostCorrespondence.lean, and the held-out Q4_K_M reference. Emitted after W15 by d1011_fields().
D1011 = {
    "vladder_1011": "docs/evidence/receipts/onebit/2026-10-11-vladder.json",
    "ray_1011": "docs/evidence/receipts/onebit/2026-10-11-ray.json",
    "kfac_1011": "docs/evidence/receipts/onebit/2026-10-11-kfac.json",
    "fold_transfer_1011": "docs/evidence/receipts/onebit/2026-10-11-fold-transfer.json",
}

RAY_ARMS = [("iq1S", "iq1_s"), ("iq1M", "iq1_m"), ("q4KM", "q4_k_m"), ("hadGptq1", "had_gptq1"), ("rtn", "rtn")]
KFAC_ARMS = [("iq1S", "iq1_s"), ("iq1M", "iq1_m"), ("q4KM", "q4_k_m"), ("hadGptq1", "had_gptq1")]

SEEDS = ["0", "1", "2"]
PER_SEED_RAT = [("klHeldout", "kl_heldout"), ("klInsample", "kl_insample"),
                ("top1Heldout", "top1_heldout"), ("top1Insample", "top1_insample")]
ARM_NAT = [("bits", "bits"), ("params", "params"), ("bitsSign", "bits_sign"), ("bitsR", "bits_r"),
           ("bitsC", "bits_c"), ("bitsD", "bits_d"), ("bitsCorr", "bits_corr")]
ARM_RAT = [("bpwLinears", "bpw_linears")]
NB_NAT = [("nbSweeps", "sweeps")]
NB_RAT = [("nbJSeq", "J_seq"), ("nbJNb", "J_nb"), ("nbRatio", "ratio")]


class Fail(Exception):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    # Floats stay as their decimal text; ints stay ints.
    return json.loads(path.read_text(encoding="utf-8"), parse_float=str)


def rat(text):
    if not isinstance(text, str):
        raise Fail(f"expected decimal text, got {text!r}")
    return Fraction(text)


def nat(value):
    if not isinstance(value, int) or isinstance(value, bool):
        raise Fail(f"expected integer, got {value!r}")
    return value


def lean_rat(q: Fraction) -> str:
    return f"({q.numerator} : ℚ) / {q.denominator}"


def read_ledger(root: Path) -> dict:
    rows = {}
    for line in (root / LEDGER).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id") in rows:
            raise Fail(f"ledger has duplicate id {row.get('id')}")
        rows[row.get("id")] = row
    return rows


def all_receipts():
    """(namespace id, receipt path) for every receipt, the four originals first."""
    return ([(ns, rel) for ns, (rel, _) in RECEIPTS.items()] + list(W17.items()) + list(W15.items())
            + list(D1011.items()))


def check_ledger(root: Path) -> dict:
    """Return {namespace id: sha256} after checking each receipt against its ledger row."""
    ledger = read_ledger(root)
    shas = {}
    for ns, rel in all_receipts():
        path = root / rel
        if not path.is_file():
            raise Fail(f"missing receipt {rel}")
        stem = Path(rel).stem
        row_id = f"onebit-receipt-{stem}"
        row = ledger.get(row_id)
        if row is None:
            raise Fail(f"stale: no ledger row {row_id} for {rel}")
        if row.get("path") != rel:
            raise Fail(f"stale: ledger row {row_id} names path {row.get('path')!r}, expected {rel}")
        actual = sha256_file(path)
        if row.get("sha256") != actual:
            raise Fail(f"stale: {rel} sha256 {actual} != ledger {row_id} sha256 {row.get('sha256')}")
        shas[ns] = actual
    return shas


def arm_block(data: dict, arm: str, rel: str) -> dict:
    variants = data.get("variants")
    if not isinstance(variants, dict) or arm not in variants:
        raise Fail(f"{rel}: no variants.{arm}")
    v = variants[arm]
    seeds = v.get("seeds")
    if not isinstance(seeds, dict) or sorted(seeds.keys()) != SEEDS:
        raise Fail(f"{rel}: variants.{arm}.seeds keys are not {SEEDS}")
    return v


def emit_receipt(ns: str, rel: str, arm, sha: str, data: dict) -> list:
    out = []  # (kind, name, value) with kind in str|nat|rat|bool
    out.append(("str", "sha256", sha))
    if arm is None:
        # Side-information receipt (t6): fields of the top level and of verdict.
        out.append(("rat", "fStar", rat(data["f_star"])))
        out.append(("rat", "whiteBoundBpw", rat(data["white_bound_bpw"])))
        verdict = data["verdict"]
        inc = verdict["incr_decoded_total"]
        out.append(("rat", "incrDecodedF1d", rat(inc["F1d"])))
        out.append(("rat", "incrDecodedF2d", rat(inc["F2d"])))
        out.append(("rat", "incrDecodedF3d", rat(inc["F3d"])))
        out.append(("rat", "rAFStarTotal", rat(verdict["R_a_fstar_total"])))
        for name, key in [("routeBOpen", "route_b_open"), ("routeBClosedLinear", "route_b_closed_linear"),
                          ("routeAClosed", "route_a_closed"), ("controlsPass", "controls_pass")]:
            if not isinstance(verdict[key], bool):
                raise Fail(f"{rel}: verdict.{key} is not a bool")
            out.append(("bool", name, verdict[key]))
        return out

    out.append(("str", "arm", arm))
    v = arm_block(data, arm, rel)
    for name, key in ARM_NAT:
        out.append(("nat", name, nat(v[key])))
    for name, key in ARM_RAT:
        out.append(("rat", name, rat(v[key])))
    seeds = v["seeds"]
    for name, key in PER_SEED_RAT:
        for s in SEEDS:
            out.append(("rat", f"{name}{s}", rat(seeds[s][key])))
    if arm.endswith("-nb"):
        nb = v["nb"]
        for name, key in NB_NAT:
            out.append(("nat", name, nat(nb[key])))
        for name, key in NB_RAT:
            out.append(("rat", name, rat(nb[key])))
    return out


def means_and_checks(ns: str, rel: str, arm, data: dict) -> list:
    """Lean-side means, the Python mean as theorem RHS, and component checks."""
    out = []
    if arm is None:
        return out
    v = arm_block(data, arm, rel)
    for name, key in PER_SEED_RAT:
        vals = [rat(v["seeds"][s][key]) for s in SEEDS]
        mean = sum(vals) / 3
        stats = v.get(f"{key}_stats")
        if isinstance(stats, dict) and "mean" in stats:
            if abs(float(mean) - float(stats["mean"])) > 1e-9:
                raise Fail(f"{rel}: {key} mean {float(mean)} differs from receipt {stats['mean']}")
        out.append(("mean", f"{name}Mean", (name, mean)))
    comp = [("bitsSign", "bits_sign"), ("bitsR", "bits_r"), ("bitsC", "bits_c"),
            ("bitsD", "bits_d"), ("bitsCorr", "bits_corr")]
    total = sum(nat(v[k]) for _, k in comp)
    if total != nat(v["bits"]):
        raise Fail(f"{rel}: component bits sum {total} != bits {v['bits']}")
    out.append(("components", "bitsComponents", [n for n, _ in comp]))
    return out


def render_receipt(ns: str, rel: str, arm, sha: str, data: dict) -> str:
    fields = emit_receipt(ns, rel, arm, sha, data)
    means = means_and_checks(ns, rel, arm, data)
    lines = [f"-- Receipt `{rel}`", f"namespace Receipts.{ns}", ""]
    for kind, name, value in fields:
        if kind == "str":
            lines.append(f'def {name} : String := "{value}"')
        elif kind == "nat":
            lines.append(f"def {name} : ℕ := {value}")
        elif kind == "rat":
            lines.append(f"def {name} : ℚ := {lean_rat(value)}")
        elif kind == "bool":
            lines.append(f"def {name} : Bool := {'true' if value else 'false'}")
    if means:
        lines.append("")
    for kind, name, value in means:
        if kind == "mean":
            base, mean = value
            seeds = [f"{base}{s}" for s in SEEDS]
            lines.append(f"def {name} : ℚ := ({' + '.join(seeds)}) / 3")
            lines.append(
                f"theorem {name}_eq : {name} = {lean_rat(mean)} := by\n"
                f"  norm_num [{name}, {', '.join(seeds)}]")
            lines.append("")
        elif kind == "components":
            total_name = "bits"
            sum_expr = " + ".join(value)
            lines.append(
                f"theorem bitsComponents_eq : {total_name} = {sum_expr} := by\n"
                f"  norm_num [{total_name}, {', '.join(value)}]")
            lines.append("")
    lines.append(f"end Receipts.{ns}")
    return "\n".join(lines) + "\n"


def int_value(v) -> int:
    """A count from JSON: an int, or decimal text with no fractional part (n_cal is written 25.0)."""
    if isinstance(v, bool):
        raise Fail(f"expected a count, got {v!r}")
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        q = Fraction(v)
        if q.denominator != 1:
            raise Fail(f"expected a count, got {v!r}")
        return int(q)
    raise Fail(f"expected a count, got {v!r}")


def rat_value(v) -> Fraction:
    return Fraction(v) if isinstance(v, int) and not isinstance(v, bool) else rat(v)


def w17_fields(ns: str, rel: str, sha: str, data: dict) -> list:
    """(kind, name, value) for one W17-C receipt; the same kinds as emit_receipt."""
    out = [("str", "sha256", sha)]
    if ns == "w17_c_fit":
        for i in range(3):
            out.append(("nat", f"nCal{i}", int_value(data["n_cal"][i])))
        for name, key in [("klHeldout", "kl_heldout"), ("klInsample", "kl_insample"),
                          ("EGen", "E_gen"), ("G", "G")]:
            for i in range(3):
                out.append(("rat", f"{name}{i}", rat_value(data[key][i])))
        out.append(("rat", "alphaLoglogFit", rat_value(data["alpha_loglog_fit"])))
        out.append(("rat", "prefactor", rat_value(data["prefactor"])))
        for i in range(2):
            out.append(("rat", f"localExponents{i}", rat_value(data["local_exponents"][i])))
    elif ns == "w17_c_measure100":
        agg = data["aggregate"]
        for name, key in [("O", "O"), ("OFs", "O_fs"), ("OShift", "O_shift"),
                          ("ODown", "O_down"), ("O896Wide", "O_896_wide")]:
            out.append(("rat", name, rat_value(agg[key])))
        sw = data["sandwich"]
        out.append(("nat", "checkedPairs", int_value(sw["checked_pairs"])))
        out.append(("nat", "violations", int_value(sw["violations"])))
        qt = data["QT"]
        out.append(("rat", "Q", rat_value(qt["Q"])))
        out.append(("rat", "TDerived", rat_value(qt["T_derived"])))
        out.append(("rat", "QTO", rat_value(qt["O"])))
        out.append(("rat", "QTG", rat_value(qt["G"])))
        out.append(("rat", "tightnessJw", rat_value(data["per_kind"]["down"]["tightness_Jw"])))
    elif ns in ("w17_c_run25", "w17_c_run50", "w17_c_run100"):
        out.append(("rat", "klHeldout", rat_value(data["kl_heldout"])))
        out.append(("rat", "klInsample", rat_value(data["kl_insample"])))
        out.append(("rat", "EGen", rat_value(data["E_gen"])))
        out.append(("rat", "G", rat_value(data["G"])))
        out.append(("nat", "bits", int_value(data["bits"])))
    elif ns == "w17_d_tilt":
        sp = data["split"]
        for name in ["J_out", "J_in", "tilt_in", "tilt_out", "q2_over_J_in", "q2_over_J_out",
                     "Q_tilt", "Q_lin", "Q_product", "Q_receipt"]:
            out.append(("rat", name, rat_value(sp[name])))
        for side in ["in", "out"]:
            s = data["sets"][side]
            out.append(("rat", f"kl_{side}", rat_value(s["kl_mean"])))
            out.append(("rat", f"q2_{side}", rat_value(s["q2_mean"])))
            out.append(("nat", f"positions_{side}", int_value(s["n_pos"])))
            out.append(("rat", f"kl_share_range8_{side}", rat_value(s["kl_share_range_gt_8"])))
        out.append(("bool", "pass_", bool_value(data["pass_"], ns)))
        for name, key in [("controlDeltaZeroPass", "delta_zero"), ("controlEps005Pass", "eps_0.05"),
                          ("controlRun100Pass", "run100_reproduced")]:
            block = data["controls"].get(key)
            if block is not None:
                out.append(("bool", name, bool_value(block["pass_"], ns)))
    elif ns in D1011:
        return d1011_fields(ns, rel, sha, data)
    else:
        return w15_fields(ns, rel, sha, data)
    return out


def d1011_fields(ns: str, rel: str, sha: str, data: dict) -> list:
    """(kind, name, value) for the 2026-10-11 escalation receipts; the same kinds as emit_receipt."""
    out = [("str", "sha256", sha)]
    if ns == "vladder_1011":
        sep, orc = data["models"]["sep"], data["models"]["oracle"]
        out.append(("rat", "sepKL1133", rat_value(sep["KL_at_1.133"])))
        out.append(("rat", "sepRealizedKL1133", rat_value(sep["realized_KL_at_1.133"])))
        out.append(("rat", "oracleKL1133", rat_value(orc["KL_at_1.133"])))
        out.append(("rat", "q4KLTotal", rat_value(data["q4_kl_total"])))
        out.append(("rat", "sepRStarAllin", rat_value(sep["R_star_allin"])))
        out.append(("rat", "oracleRStarAllin", rat_value(orc["R_star_allin"])))
    elif ns == "ray_1011":
        for name, key in RAY_ARMS:
            kl = data["arms"][key]["kl"]
            out.append(("rat", f"{name}Kl0125", rat_value(kl["0.125"])))
            out.append(("rat", f"{name}Kl1", rat_value(kl["1.0"])))
        out.append(("rat", "hadGptq1KlNeg1", rat_value(data["arms"]["had_gptq1"]["kl"]["-1.0"])))
    elif ns == "kfac_1011":
        for name, key in KFAC_ARMS:
            v = data["validation"][key]
            out.append(("rat", f"kfacDown{name}", rat_value(v["kfac_down"])))
            out.append(("rat", f"exactQuadratic{name}", rat_value(v["exact_quadratic"])))
        sep, orc = data["floor"]["separable"], data["floor"]["oracle"]
        out.append(("rat", "sepKl118", rat_value(sep["by_rate"]["1.18"]["kl"])))
        out.append(("rat", "oracleKl118", rat_value(orc["by_rate"]["1.18"]["kl"])))
        out.append(("rat", "q4KKfacAll", rat_value(data["q4_k_m_kfac_all"])))
        out.append(("rat", "sepRateAtQ4", rat_value(sep["rate_bpw_at_q4_kl"])))
        out.append(("rat", "oracleRateAtQ4", rat_value(orc["rate_bpw_at_q4_kl"])))
    elif ns == "fold_transfer_1011":
        q = data["variants"]["q4km"]
        out.append(("str", "arm", "q4km"))
        out.append(("rat", "q4kmKlHeldout", rat_value(q["kl_heldout"])))
        out.append(("rat", "q4kmKlInsample", rat_value(q["kl_insample"])))
        fl, rt = data["variants"]["fold-local"], data["variants"]["rtn"]
        out.append(("rat", "foldLocalKlHeldout", rat_value(fl["kl_heldout"])))
        out.append(("rat", "foldLocalBpw", rat_value(fl["bpw_linears"])))
        out.append(("rat", "rtnKlHeldout", rat_value(rt["kl_heldout"])))
    else:
        raise Fail(f"no 2026-10-11 field map for {ns}")
    return out


BACKFIT_ARM = "had-fold-2plane-attn-backfit"


def w15_fields(ns: str, rel: str, sha: str, data: dict) -> list:
    """(kind, name, value) for the W15 receipts; the same kinds as emit_receipt."""
    out = [("str", "sha256", sha)]
    if ns == "w15_backfit_attn":
        # Single-seed arm: no seeds block.
        v = data["variants"][BACKFIT_ARM]
        out.append(("str", "arm", BACKFIT_ARM))
        out.append(("rat", "klHeldout", rat(v["kl_heldout"])))
        out.append(("rat", "klInsample", rat(v["kl_insample"])))
    elif ns == "t3_base_seeds12":
        seeds = data["variants"][T0_ARM]["seeds"]
        if sorted(seeds.keys()) != ["1", "2"]:
            raise Fail(f"{rel}: seeds keys are not ['1', '2']")
        out.append(("str", "arm", T0_ARM))
        out.append(("rat", "klHeldout1", rat(seeds["1"]["kl_heldout"])))
        out.append(("rat", "klHeldout2", rat(seeds["2"]["kl_heldout"])))
    elif ns == "t6_sideinfo_ext":
        res = data["result"]
        out.append(("rat", "incrOverF0F2dUp", rat(res["incr_over_F0"]["F2d"]["up"])))
        out.append(("rat", "incrOverF0F2oUp", rat(res["incr_over_F0"]["F2o"]["up"])))
        out.append(("rat", "shuffleF1KDR2", rat(res["shuffle_delta_r2"]["F1"]["per_kind"]["k"])))
    elif ns == "w18_outcomes":
        for pre, key in OUTCOME_ARMS:
            o = data["variants"][key]["outcomes"]
            for name, k in OUTCOME_RAT:
                out.append(("rat", pre + name, rat(o[k])))
            out.append(("rat", pre + "Kl", rat(data["variants"][key]["kl_heldout"])))
            out.append(("nat", pre + "ArcN", nat(o["arc_n"])))
            out.append(("nat", pre + "HsN", nat(o["hs_n"])))
        gens = data["variants"][RESQ_ARM]["outcomes"]["generation"]
        for i, g in enumerate(gens):
            out.append(("nat", f"resqFirstDivergence{i}", nat(g["first_divergence"])))
    elif ns in ("w18_kd_main1", "w18_kd_main2"):
        out.append(("nat", "steps", nat(data["args"]["steps"])))
        out.append(("nat", "tokensTrained", nat(data["tokens_trained"])))
        out.append(("nat", "params", nat(data["linear_params"])))
        out.append(("nat", "bits", nat(data["bits"])))
        for name, k in [("bpwLinears", "bpw_linears"), ("klInit", "kl_heldout_init"), ("klHeldout", "kl_heldout"),
                        ("top1Heldout", "top1_heldout"), ("signFlipFrac", "sign_flip_frac"),
                        ("trainWallS", "train_wall_s")]:
            out.append(("rat", name, rat(data[k])))
    elif ns == "w18_calx600":
        v = data["variants"][RESQ_ARM]
        out.append(("str", "arm", RESQ_ARM))
        out.append(("nat", "calTokens", nat(data["cal_tokens"])))
        out.append(("nat", "calExtraChunks", nat(data["args"]["cal_extra"])))
        out.append(("nat", "params", nat(v["params"])))
        out.append(("nat", "bits", nat(v["bits"])))
        for name, k in [("bpwLinears", "bpw_linears"), ("klHeldout", "kl_heldout"), ("klInsample", "kl_insample"),
                        ("top1Heldout", "top1_heldout"), ("top1Insample", "top1_insample"),
                        ("wallTransferS", "wall_transfer_s")]:
            out.append(("rat", name, rat(v[k])))
    elif ns == "w18_trellis2":
        lin, sm, c = data["linears"], data["summary"], data["controls"]
        out.append(("nat", "linears", nat(len(lin))))
        for L in ("L8", "L12"):
            for b in ("beam1", "beam4"):
                key = f"{L}_vs_{b}"
                if sm[key]["n"] != len(lin):
                    raise Fail(f"{rel}: summary {key} n != linears")
                out.append(("rat", f"median{L}{b.capitalize()}", rat(sm[key]["median"])))
                out.append(("rat", f"geomean{L}{b.capitalize()}", rat(sm[key]["geomean"])))
                # Minimum over every linear: the per-linear ratios, not a summary field.
                out.append(("rat", f"min{L}{b.capitalize()}", rat(min(x[f"ratio_{L}_vs_{b}"] for x in lin))))
        dg = c["degenerate"]
        for name, k in [("signL8Ratio", "ratio"), ("signL1Ratio", "ratio_L1"), ("gptqSignRatio", "ratio_gptq_sign")]:
            out.append(("rat", name, rat(dg[k])))
        out.append(("rat", "identityRel", rat(c["identity"]["rel"])))
        out.append(("rat", "wallS", rat(data["total_wall"])))
    elif ns in ("w18_size05", "w18_size15", "w18_size15m"):
        # Size trend: the arm and the model's own Q4_K_M on one protocol (50 calibration chunks, held-out [50, 90)).
        # w18_size15m is the arm alone with the gate/up second plane widened to 792 rows (stored bits matched to 0.5B).
        v = data["variants"][RESQ_ARM]
        if data["args"]["n_cal"] != 50 or data["eval_chunks"] != [50, 90]:
            raise Fail(f"{rel}: not the n-cal 50 / held-out [50, 90) protocol")
        if data["args"].get("qmlp_rows", 416) != (792 if ns == "w18_size15m" else 416):
            raise Fail(f"{rel}: unexpected qmlp_rows")
        out.append(("nat", "qmlpRows", nat(data["args"].get("qmlp_rows", 416))))
        out.append(("str", "arm", RESQ_ARM))
        out.append(("str", "model", data["args"]["model"]))
        out.append(("nat", "layers", nat(data["args"]["layers"])))
        out.append(("nat", "linearParams", nat(data["linear_params"])))
        out.append(("nat", "params", nat(v["params"])))
        out.append(("nat", "bits", nat(v["bits"])))
        out.append(("nat", "bitsSign", nat(v["bits_sign"])))
        for name, k in [("bpwLinears", "bpw_linears"), ("klHeldout", "kl_heldout"), ("klInsample", "kl_insample"),
                        ("top1Heldout", "top1_heldout"), ("wallTransferS", "wall_transfer_s")]:
            out.append(("rat", name, rat(v[k])))
        if ns != "w18_size15m":
            q = data["variants"]["q4km"]
            for name, k in [("q4kmBpw", "bpw_linears"), ("q4kmKlHeldout", "kl_heldout"), ("q4kmTop1Heldout", "top1_heldout")]:
                out.append(("rat", name, rat(q[k])))
        out.append(("rat", "wallTotalS", rat(data["wall_total_s"])))
    elif ns in ("w18_pvste05", "w18_pvpre05"):
        # End-to-end finishing with the E8 u O8 codes trained by a straight-through projection (pv_finish.py --codes ste).
        # pvste05 starts each latent at its codeword under Adam; pvpre05 at the plane's continuous pre-image under nsgd.
        if data["codes"] != "ste" or data["v_steps"]:
            raise Fail(f"{rel}: not a straight-through code run, or V-steps ran")
        pre = ns == "w18_pvpre05"
        if (data.get("code_init", "codeword"), data.get("code_opt", "adam")) != (("preimage", "nsgd") if pre else ("codeword", "adam")):
            raise Fail(f"{rel}: code init / optimiser is not the one this namespace records")
        if pre:
            if data["step0_projection_mismatch"] != 0 or not data["controls"]["step0_projection_mismatch_zero"]:
                raise Fail(f"{rel}: step-0 projection differs from the stored code")
            last = data["code_changes"][-1]
            if last["changed_cum"] != data["code_changed_total"]:
                raise Fail(f"{rel}: last code-change record is not the total")
            out.append(("nat", "step0ProjectionMismatch", nat(data["step0_projection_mismatch"])))
            out.append(("rat", "preimageNearestIsIdx", rat(data["preimage_nearest_is_idx"])))
            out.append(("rat", "preimageClipped", rat(data["preimage_clipped"])))
            out.append(("nat", "flipsTowardPreimage", nat(last["flips_toward_preimage_cum_count"])))
        if data["arm"] != RESQ_ARM or data["n_cal"] != 50 or data["eval_chunks"] != [50, 90] or data["insample_chunks"] != [0, 20]:
            raise Fail(f"{rel}: not the arm on the n-cal 50 / held-out [50, 90) protocol")
        c = data["controls"]
        if c["quantized_eval_equality"]["linears_not_equal"] != 0 or c["data_disjoint"]["overlap"] != 0:
            raise Fail(f"{rel}: eval weights are not the stored code, or train overlaps calibration")
        bc = data["bits_components"]
        if bc != data["bits_components_after"] or sum(bc.values()) != data["bits_before"]:
            raise Fail(f"{rel}: bit components changed or do not sum to the stored bits")
        out.append(("str", "arm", RESQ_ARM))
        out.append(("bool", "beforeMatchesReference", bool_value(c["before_kl_regression"]["ok"], ns)))
        out.append(("bool", "bitsEqual", bool_value(c["bits_equal"]["ok"], ns)))
        out.append(("bool", "codebookExact", bool_value(c["codebook_match"]["ok"], ns)))
        out.append(("bool", "dataDisjoint", bool_value(c["data_disjoint"]["ok"], ns)))
        for name, k in [("params", "params"), ("bitsBefore", "bits_before"), ("bitsAfter", "bits_after"),
                        ("steps", "steps"), ("tokens", "tokens"), ("codePlanes", "code_planes"),
                        ("codeBlocks", "code_blocks"), ("codeChangedTotal", "code_changed_total")]:
            out.append(("nat", name, nat(data[k])))
        out.append(("nat", "codeKinds", nat(len(data["codes_trained_kinds"]))))
        out.append(("rat", "lrCode", rat(data["lr_code"])))
        # The first 25-step window: blocks moved, and the train loss at step 1 against step 25.
        w0 = data["code_changes"][0]
        if w0["step"] != 25 or data["train_trajectory"][0]["step"] != 1:
            raise Fail(f"{rel}: first code-change window is not steps 1..25")
        out.append(("nat", "codeChangedFirstWindow", nat(w0["changed_window"])))
        out.append(("rat", "trainLossStep1", rat(data["train_trajectory"][0]["train_loss"])))
        out.append(("rat", "trainLossStep25", rat(w0["train_loss"])))
        out.append(("rat", "trainLossLast", rat(data["train_loss_last"])))
        for name, k in [("klHeldoutBefore", "kl_heldout_before"), ("klHeldoutAfter", "kl_heldout_after"),
                        ("top1HeldoutBefore", "top1_heldout_before"), ("top1HeldoutAfter", "top1_heldout_after"),
                        ("klInsampleBefore", "kl_insample_before"), ("klInsampleAfter", "kl_insample_after")]:
            out.append(("rat", name, rat(data[k])))
        out.append(("rat", "wallTotalS", rat(data["walls"]["total_s"])))
    elif ns in ("w18_pv05", "w18_pvobs05", "w18_pvlam05"):
        # End-to-end KL finishing of the arm's stored code (pv_finish.py): the same arm, before and after, at identical bits.
        # pv05 scores V-step candidates by the linear gain; pvobs05 and pvlam05 by the diagonal-Fisher OBS gain.
        obs = ns != "w18_pv05"
        if data.get("v_gain", "linear") != ("obs" if obs else "linear"):
            raise Fail(f"{rel}: V-step gain is not the one this namespace records")
        if data["arm"] != RESQ_ARM or data["n_cal"] != 50 or data["eval_chunks"] != [50, 90] or data["insample_chunks"] != [0, 20]:
            raise Fail(f"{rel}: not the arm on the n-cal 50 / held-out [50, 90) protocol")
        c = data["controls"]
        if c["pack_equality"]["torch_equal"] != c["pack_equality"]["linears"]:
            raise Fail(f"{rel}: pack sink does not reconstruct every linear")
        if c["quantized_eval_equality"]["linears_not_equal"] != 0 or c["data_disjoint"]["overlap"] != 0:
            raise Fail(f"{rel}: eval weights are not the stored code, or train overlaps calibration")
        out.append(("str", "arm", RESQ_ARM))
        out.append(("bool", "beforeMatchesReference", bool_value(c["before_kl_regression"]["ok"], ns)))
        out.append(("bool", "bitsEqual", bool_value(c["bits_equal"]["ok"], ns)))
        out.append(("bool", "codebookExact", bool_value(c["codebook_match"]["ok"], ns)))
        out.append(("bool", "dataDisjoint", bool_value(c["data_disjoint"]["ok"], ns)))
        for name, k in [("params", "params"), ("bitsBefore", "bits_before"), ("bitsAfter", "bits_after"),
                        ("steps", "steps"), ("tokens", "tokens")]:
            out.append(("nat", name, nat(data[k])))
        out.append(("nat", "vSteps", nat(len(data["v_steps"]))))
        out.append(("nat", "vAccepted", nat(sum(1 for x in data["v_steps"] if x["accepted"]))))
        # Every V-step candidate (skip-rank x fraction) against the loss it realised on its own scoring batch.
        tries = [(t, x["loss_before"]) for x in data["v_steps"] for t in x["tries"]]
        fell = [(t, b) for t, b in tries if rat(t["loss_after"]) < rat(b)]
        out.append(("nat", "vTries", nat(len(tries))))
        out.append(("nat", "vTriesPredictedFall", nat(sum(1 for t, _ in tries if rat(t["predicted_dloss"]) < 0))))
        out.append(("nat", "vTriesFell", nat(len(fell))))
        if obs:
            if fell:
                raise Fail(f"{rel}: expected no OBS V-step candidate to lower the loss, got {len(fell)}")
            lams = sorted({rat(t["lam"]) for t, _ in tries})
            out.append(("rat", "obsLamMin", lams[0]))
            out.append(("rat", "obsLamMax", lams[-1]))
            # The candidate with the smallest realised rise, and the one with the largest realised/|predicted| ratio floor.
            t, b = min(tries, key=lambda x: rat(x[0]["loss_after"]) - rat(x[1]))
            out.append(("rat", "vBestLam", rat(t["lam"])))
            out.append(("nat", "vBestMoves", nat(t["moves"])))
            out.append(("rat", "vBestPredicted", rat(t["predicted_dloss"])))
            out.append(("rat", "vBestRealised", rat(t["loss_after"]) - rat(b)))
            # At the largest lam, the smallest realised rise over |predicted| across every try.
            top = [(rat(t["loss_after"]) - rat(b)) / -rat(t["predicted_dloss"]) for t, b in tries if rat(t["lam"]) == lams[-1]]
            out.append(("rat", "vRiseOverPredMinAtLamMax", min(top)))
        else:
            if len(fell) != 1:
                raise Fail(f"{rel}: expected exactly one V-step candidate to lower the loss, got {len(fell)}")
            t, b = fell[0]
            out.append(("nat", "vFellMoves", nat(t["moves"])))
            out.append(("rat", "vFellPredicted", rat(t["predicted_dloss"])))
            out.append(("rat", "vFellRealised", rat(t["loss_after"]) - rat(b)))
        bc = data["bits_components"]
        if bc != data["bits_components_after"] or sum(bc.values()) != data["bits_before"]:
            raise Fail(f"{rel}: bit components changed or do not sum to the stored bits")
        out.append(("nat", "bitsSign", nat(bc["bits_sign"])))
        out.append(("nat", "bitsScales", nat(bc["bits_r"] + bc["bits_c"] + bc["bits_d"])))
        out.append(("nat", "bitsCorr", nat(bc["bits_corr"])))
        for name, k in [("klHeldoutBefore", "kl_heldout_before"), ("klHeldoutAfter", "kl_heldout_after"),
                        ("top1HeldoutBefore", "top1_heldout_before"), ("top1HeldoutAfter", "top1_heldout_after"),
                        ("klInsampleBefore", "kl_insample_before"), ("klInsampleAfter", "kl_insample_after")]:
            out.append(("rat", name, rat(data[k])))
        out.append(("rat", "wallTotalS", rat(data["walls"]["total_s"])))
    elif ns in PVAB_GROUPS:
        # Finishing with only a subset of the zero-bit-cost tunables trained (pv_finish.py --train): r = group scales,
        # c, d = column scales, norm = norms and q/k/v biases. Codes are fixed: V-steps off, no straight-through codes.
        if data["codes"] != "vstep" or data["args"]["v_every"] != 0 or data["v_steps"]:
            raise Fail(f"{rel}: V-steps ran, or the codes were trained")
        if data["train_groups"] != PVAB_GROUPS[ns]:
            raise Fail(f"{rel}: trained groups {data['train_groups']} are not {PVAB_GROUPS[ns]}")
        if data["arm"] != RESQ_ARM or data["n_cal"] != 50 or data["eval_chunks"] != [50, 90] or data["insample_chunks"] != [0, 20]:
            raise Fail(f"{rel}: not the arm on the n-cal 50 / held-out [50, 90) protocol")
        c = data["controls"]
        if c["pack_equality"]["torch_equal"] != c["pack_equality"]["linears"]:
            raise Fail(f"{rel}: pack sink does not reconstruct every linear")
        if c["quantized_eval_equality"]["linears_not_equal"] != 0 or c["data_disjoint"]["overlap"] != 0:
            raise Fail(f"{rel}: eval weights are not the stored code, or train overlaps calibration")
        bc = data["bits_components"]
        if bc != data["bits_components_after"] or sum(bc.values()) != data["bits_before"]:
            raise Fail(f"{rel}: bit components changed or do not sum to the stored bits")
        out.append(("str", "arm", RESQ_ARM))
        out.append(("str", "trainGroups", ",".join(data["train_groups"])))
        out.append(("bool", "beforeMatchesReference", bool_value(c["before_kl_regression"]["ok"], ns)))
        out.append(("bool", "bitsEqual", bool_value(c["bits_equal"]["ok"], ns)))
        out.append(("bool", "codebookExact", bool_value(c["codebook_match"]["ok"], ns)))
        out.append(("bool", "dataDisjoint", bool_value(c["data_disjoint"]["ok"], ns)))
        for name, k in [("params", "params"), ("bitsBefore", "bits_before"), ("bitsAfter", "bits_after"),
                        ("steps", "steps"), ("tokens", "tokens"), ("trainableParams", "trainable_params")]:
            out.append(("nat", name, nat(data[k])))
        out.append(("nat", "vSteps", nat(len(data["v_steps"]))))
        if "regrid" in data:
            # Zero-bit r-grid refit every `regrid` steps (0 = off) and the step cap that ends the cosine schedule.
            out.append(("nat", "regrid", nat(data["regrid"])))
            out.append(("nat", "stepsCap", nat(min(data["args"]["steps"], 10 ** 9))))
            if data["regrid"]:
                out.append(("rat", "rSatFracBefore", rat(data["r_sat_frac_before"])))
                out.append(("rat", "rSatFracAfter", rat(data["r_sat_frac_after"])))
        if (data["args"].get("compose", "layer"), data["args"].get("kl", "ckpt")) != ("layer", "ckpt"):
            # Throughput path (PlaneCompose once per step, KLHead): the start-of-training probe compares losses and the
            # trainable gradient against the (layer, ckpt) path, with a second (layer, ckpt) pass as the noise floor.
            # Arms on the fwht128 butterfly must be loss-bit-equal.  Under --hadamard mm (summation order differs) the mm
            # arithmetic is gated in float64 on real planes (PlaneComposeMM vs PlaneCompose, forward and adjoints, max
            # relative < 1e-12); at model level the fp32-student mm arm against the fp32 butterfly arm holds loss relative
            # < 1e-5 and gradient relative L2 < 1e-2, the bf16 mm arms gradient relative L2 < 1e-1.
            ge = c["grad_equiv"]
            had = data["args"].get("hadamard", "butterfly")
            mm_arms = {"had_only", "requested"} if had == "mm" else set()
            for k, v in ge["arms"].items():
                if k in mm_arms:
                    fh = ge["fp32"]["fp32_had"]
                    ok_k = (rat(ge["fp32"]["float64_unit_max_rel"]) < Fraction(1, 10 ** 12) and rat(fh["loss_rel"]) < Fraction(1, 10 ** 5)
                            and rat(fh["grad_rel_l2"]) < Fraction(1, 100) and rat(v["grad_rel_l2"]) < Fraction(1, 10))
                else:
                    ok_k = k == "jitter" or v["loss_equal"]
                if not ok_k:
                    raise Fail(f"{rel}: grad_equiv arm {k} outside its tolerance")
            if not ge["ok"]:
                raise Fail(f"{rel}: grad_equiv control failed")
            out.append(("str", "compose", data["args"]["compose"]))
            out.append(("str", "klMode", data["args"]["kl"]))
            out.append(("str", "hadamard", had))
            out.append(("bool", "gradEquivOk", bool_value(ge["ok"], ns)))
            arms = [("gradRelFloor", "ref_again"), ("gradRelComposeOnly", "compose_only"), ("gradRelKlOnly", "kl_only"),
                    ("gradRelRequested", "requested")]
            if had == "mm":
                arms += [("gradRelHadOnly", "had_only"), ("gradRelJitter", "jitter")]
            for name, k in arms:
                out.append(("rat", name, rat(ge["arms"][k]["grad_rel_l2"])))
            if had == "mm":
                for name, k in [("lossRelHadOnly", "had_only"), ("lossRelJitter", "jitter"), ("lossRelRequested", "requested")]:
                    out.append(("rat", name, rat(ge["arms"][k]["loss_rel"])))
                for name, k in [("bf16FlipsMm", "mm"), ("bf16FlipsJitter", "jitter")]:
                    out.append(("nat", name, nat(ge["weights"][k]["bf16_flips"])))
                out.append(("rat", "wRelMaxMm", rat(ge["weights"]["mm"]["w_rel_max"])))
                for name, k in [("bf16RelL2Mm", "mm"), ("bf16RelL2Jitter", "jitter")]:
                    out.append(("rat", name, rat(ge["weights"][k]["bf16_rel_l2"])))
                out.append(("nat", "zeroToNonzeroMm", nat(ge["weights"]["mm"]["zero_to_nonzero"])))
                fp = ge["fp32"]
                out.append(("rat", "lossRelFp32Had", rat(fp["fp32_had"]["loss_rel"])))
                out.append(("rat", "gradRelFp32Had", rat(fp["fp32_had"]["grad_rel_l2"])))
                out.append(("rat", "gradRelFp32Floor", rat(fp["fp32_ref_again"]["grad_rel_l2"])))
                out.append(("rat", "lossRelBf16VsFp32", rat(fp["bf16_vs_fp32_loss_rel"])))
                out.append(("rat", "float64UnitMaxRel", rat(fp["float64_unit_max_rel"])))
            out.append(("rat", "peakMemGb", rat(data["peak_mem_gb"])))
        out.append(("rat", "trainS", rat(data["walls"]["train_s"])))
        for name, k in [("bitsR", "bits_r"), ("bitsC", "bits_c"), ("bitsD", "bits_d")]:
            out.append(("nat", name, nat(bc[k])))
        for name, k in [("klHeldoutBefore", "kl_heldout_before"), ("klHeldoutAfter", "kl_heldout_after"),
                        ("top1HeldoutBefore", "top1_heldout_before"), ("top1HeldoutAfter", "top1_heldout_after"),
                        ("klInsampleBefore", "kl_insample_before"), ("klInsampleAfter", "kl_insample_after")]:
            out.append(("rat", name, rat(data[k])))
        out.append(("rat", "wallTotalS", rat(data["walls"]["total_s"])))
    elif ns in PROFILES:
        if data["n_cal"] != 50 or data["args"]["v_every"] != 0 or data["codes"] != "vstep":
            raise Fail(f"{rel}: not the n-cal 50, V-steps-off finishing protocol")
        if (data["args"].get("compose", "layer"), data["args"].get("kl", "ckpt")) != PROFILES[ns]:
            raise Fail(f"{rel}: path is not {PROFILES[ns]}")
        pr, ops = data["profile"], data["profile"]["ops"]
        aten = [o for o in ops if o["name"].startswith("aten::")]
        out.append(("str", "arm", RESQ_ARM))
        out.append(("nat", "steps", nat(pr["steps"])))
        out.append(("rat", "wallS", rat(pr["wall_s"])))
        out.append(("rat", "selfDeviceS", rat(pr["self_device_s"])))
        out.append(("rat", "atenSelfS", sum((rat(o["self_device_s"]) for o in aten), Fraction(0))))
        out.append(("rat", "elementwiseS", sum((rat(o["self_device_s"]) for o in aten if o["name"] in PROFILE_ELEMENTWISE),
                                               Fraction(0))))
        out.append(("rat", "mmS", sum((rat(o["self_device_s"]) for o in aten if o["name"] == "aten::mm"), Fraction(0))))
        out.append(("nat", "catCalls", nat(sum(o["calls"] for o in aten if o["name"] == "aten::cat"))))
    elif ns == "qreg_kan_probe":
        # Learned register angles vs fwht128 (kan_probe.py): J sums over the four input sites of every layer.
        rows, a = data["rows"], data["args"]
        if a["n_cal"] != 100 or a["n_eval"] != 40 or a["angle_bits"] != 16 or len(rows) != 96:
            raise Fail(f"{rel}: not the 100/40-chunk, 16-bit, 24-layer x 4-site protocol")
        out.append(("nat", "rows", nat(len(rows))))
        out.append(("nat", "angleBitsFullModel", nat(data["angle_bits_full_model"])))
        out.append(("nat", "linearParams", nat(data["linear_params"])))
        for name, k in [("identityMaxAbs", "identity_max_abs"), ("ratioCal", "ratio_cal"), ("geoCal", "geo_cal"),
                        ("ratioEv", "ratio_ev"), ("geoEv", "geo_ev"), ("wallS", "wall_s")]:
            out.append(("rat", name, rat(data[k])))
        for site, pre in [("attn_in", "attnIn"), ("attn_out_in", "attnOutIn"), ("ffn_in", "ffnIn"),
                          ("ffn_down_in", "ffnDownIn")]:
            rs = [r for r in rows if r["site"] == site]
            for split, cap in [("cal", "Cal"), ("ev", "Ev")]:
                out.append(("rat", pre + "JHad" + cap, sum((rat(r[f"J_had_{split}"]) for r in rs), Fraction(0))))
                out.append(("rat", pre + "JKan" + cap, sum((rat(r[f"J_kan_{split}"]) for r in rs), Fraction(0))))
    elif ns in GUIDED:
        # End-loss-weighted Grams (fold_transfer "-guided[a|b]NN"): per variant, prefix `base` for the arm itself and
        # `g<part><NN>` otherwise (part ab = token and row weights, a = token only, b = row only; NN = 100 * gamma).
        if data["args"]["n_cal"] != 50 or data["eval_chunks"] != [50, 90] or data["insample_chunks"] != [0, 20]:
            raise Fail(f"{rel}: not the n-cal 50 / held-out [50, 90) protocol")
        names = sorted(data["variants"])
        if sorted(guided_prefix(n, rel) for n in names) != sorted(GUIDED[ns]):
            raise Fail(f"{rel}: variants {names} are not {GUIDED[ns]}")
        out.append(("str", "arm", RESQ_ARM))
        seeds = GUIDED_SEEDS.get(ns)
        if data["args"]["seeds"] != (",".join(seeds) if seeds else "0"):
            raise Fail(f"{rel}: seeds {data['args']['seeds']} are not {seeds or ['0']}")
        for n in names:
            v, pre = data["variants"][n], guided_prefix(n, rel)
            out.append(("nat", pre + "Bits", nat(v["bits"])))
            for name, k in [("KlHeldout", "kl_heldout"), ("KlInsample", "kl_insample"), ("Top1Heldout", "top1_heldout")]:
                if seeds:
                    for sd in seeds:
                        out.append(("rat", pre + name + "S" + sd, rat(v["seeds"][sd][k])))
                else:
                    out.append(("rat", pre + name, rat(v[k])))
            if pre != "base":
                g = v["gq"]
                out.append(("rat", pre + "Gamma", rat(g["gamma"])))
                out.append(("str", pre + "Part", g.get("part", "ab")))
                out.append(("nat", pre + "CaptureK", nat(g["K"])))
                # Kish effective sample size of the token weights, as a fraction of the tokens: min over layers and stages.
                out.append(("rat", pre + "EssMin", rat(min(g["ess_frac"].values()))))
    else:
        raise Fail(f"no W15/W17 field map for {ns}")
    return out


def guided_prefix(name: str, rel: str) -> str:
    """`base` for RESQ_ARM, `g<part><NN>` for RESQ_ARM + "-guided[a|b]NN" (bare NN = 100)."""
    if name == RESQ_ARM:
        return "base"
    m = re.fullmatch(re.escape(RESQ_ARM) + r"-guided([ab])?(\d+)?", name)
    if not m:
        raise Fail(f"{rel}: variant {name} is neither the arm nor a -guided arm")
    return "g" + (m.group(1) or "ab") + (m.group(2) or "100")


def bool_value(v, ns: str) -> bool:
    if not isinstance(v, bool):
        raise Fail(f"{ns}: expected a bool, got {v!r}")
    return v


def render_w17(ns: str, rel: str, sha: str, data: dict) -> str:
    lines = [f"-- Receipt `{rel}`", f"namespace Receipts.{ns}", ""]
    for kind, name, value in w17_fields(ns, rel, sha, data):
        if kind == "str":
            lines.append(f'def {name} : String := "{value}"')
        elif kind == "nat":
            lines.append(f"def {name} : ℕ := {value}")
        elif kind == "rat":
            lines.append(f"def {name} : ℚ := {lean_rat(value)}")
        elif kind == "bool":
            lines.append(f"def {name} : Bool := {'true' if value else 'false'}")
    lines.append("")
    lines.append(f"end Receipts.{ns}")
    return "\n".join(lines) + "\n"


def generate(root: Path) -> str:
    shas = check_ledger(root)
    head = [
        "-- GENERATED by tools/rooting/receipt_lean.py; do not edit",
        "-- Source: docs/evidence/receipts/onebit/*.json (ledger rows in docs/evidence/ledger.jsonl).",
        "-- Values are exact rationals from the JSON decimal text. Means are defined here and checked",
        "-- against the Python mean by norm_num.",
        "import Mathlib.Tactic.NormNum",
        "",
    ]
    body = []
    for ns, (rel, arm) in RECEIPTS.items():
        data = load_json(root / rel)
        body.append(render_receipt(ns, rel, arm, shas[ns], data))
    for ns, rel in list(W17.items()) + list(W15.items()) + list(D1011.items()):
        data = load_json(root / rel)
        body.append(render_w17(ns, rel, shas[ns], data))
    return "\n".join(head) + "\n" + "\n".join(body)


def cmd_check(root: Path) -> int:
    try:
        text = generate(root)
    except Fail as e:
        print(f"receipt_lean: {e}", file=sys.stderr)
        return 1
    target = root / OUT
    if not target.is_file():
        print(f"receipt_lean: missing {OUT}", file=sys.stderr)
        return 1
    current = target.read_text(encoding="utf-8")
    if current == text:
        return 0
    cur_lines = current.splitlines()
    exp_lines = text.splitlines()
    for i in range(max(len(cur_lines), len(exp_lines))):
        a = exp_lines[i] if i < len(exp_lines) else "<EOF>"
        b = cur_lines[i] if i < len(cur_lines) else "<EOF>"
        if a != b:
            print(f"receipt_lean: drift {OUT}:{i + 1}", file=sys.stderr)
            print(f"  expected: {a}", file=sys.stderr)
            print(f"  committed: {b}", file=sys.stderr)
            return 1
    return 1


def cmd_write(root: Path) -> int:
    try:
        text = generate(root)
    except Fail as e:
        print(f"receipt_lean: {e}", file=sys.stderr)
        return 1
    target = root / OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"receipt_lean: wrote {OUT} ({len(text.splitlines())} lines)")
    return 0


def copy_fixture(dst: Path) -> None:
    (dst / "docs/evidence/receipts/onebit").mkdir(parents=True)
    shutil.copy(REPO / LEDGER, dst / LEDGER)
    for ns, rel in all_receipts():
        shutil.copy(REPO / rel, dst / rel)


def mutate(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) < 1:
        raise Fail(f"control setup: {old!r} not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def cmd_self_test() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        def fresh(name: str) -> Path:
            d = base / name
            d.mkdir()
            copy_fixture(d)
            if cmd_write(d) != 0:
                raise Fail(f"{name}: generation failed")
            return d

        # Baseline: a fresh generation checks green.
        d = fresh("baseline")
        if cmd_check(d) != 0:
            print("self-test: baseline is not green", file=sys.stderr)
            return 1
        print("self-test: baseline green")

        # (i) one digit of a generated constant.
        d = fresh("control-i")
        mutate(d / OUT, "def fStar : ℚ := (98479 : ℚ) / 100000", "def fStar : ℚ := (98480 : ℚ) / 100000")
        rc = cmd_check(d)
        print(f"self-test: control (i) generated digit -> exit {rc}")
        ok &= rc == 1

        # (ii) one digit of a committed receipt copy: sha256 is stale against the ledger.
        d = fresh("control-ii")
        mutate(d / RECEIPTS["t0b_sinkfnorm_3seed"][0], "1.3777076184749604", "1.3777076184749605")
        rc = cmd_check(d)
        print(f"self-test: control (ii) receipt digit -> exit {rc}")
        ok &= rc == 1

        # (iii) one namespace deleted.
        d = fresh("control-iii")
        text = (d / OUT).read_text(encoding="utf-8")
        start = text.index("namespace Receipts.t0_best_3seed")
        end_marker = "end Receipts.t0_best_3seed\n"
        end = text.index(end_marker, start) + len(end_marker)
        (d / OUT).write_text(text[:start] + text[end:], encoding="utf-8")
        rc = cmd_check(d)
        print(f"self-test: control (iii) namespace deleted -> exit {rc}")
        ok &= rc == 1

        # (iv) one generated count of a W17-C receipt.
        d = fresh("control-iv")
        mutate(d / OUT, "def checkedPairs : ℕ := 336", "def checkedPairs : ℕ := 337")
        rc = cmd_check(d)
        print(f"self-test: control (iv) W17-C generated count -> exit {rc}")
        ok &= rc == 1

        # (v) one generated tilt constant of the W17-D receipt.
        d = fresh("control-v")
        mutate(d / OUT, "def tilt_in : ℚ := (", "def tilt_in : ℚ := (1 + ")
        rc = cmd_check(d)
        print(f"self-test: control (v) W17-D generated tilt constant -> exit {rc}")
        ok &= rc == 1

    if not ok:
        print("self-test: a control was not red", file=sys.stderr)
        return 1
    print("self-test: 5/5 controls red")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    g.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=str(REPO))
    args = ap.parse_args(argv)
    root = Path(args.root)
    if args.self_test:
        try:
            return cmd_self_test()
        except Fail as e:
            print(f"receipt_lean: {e}", file=sys.stderr)
            return 1
    if args.check:
        return cmd_check(root)
    return cmd_write(root)


if __name__ == "__main__":
    sys.exit(main())
