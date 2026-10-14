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
}

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
            out.append(("nat", pre + "ArcN", nat(o["arc_n"])))
            out.append(("nat", pre + "HsN", nat(o["hs_n"])))
        gens = data["variants"][RESQ_ARM]["outcomes"]["generation"]
        for i, g in enumerate(gens):
            out.append(("nat", f"resqFirstDivergence{i}", nat(g["first_divergence"])))
    else:
        raise Fail(f"no W15/W17 field map for {ns}")
    return out


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
