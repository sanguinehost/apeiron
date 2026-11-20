#!/usr/bin/env python3
"""Build a targeted two-plane residual pack by splicing two packs of the same source.

The uniform residual arm fits both planes on every section and costs 1,742,045,184 B. The fit is
per-section and independent, so the pack that serves two planes on a chosen subset is a byte-level
splice of the residual-free pack (v2) and the uniform two-plane pack (v3): unselected sections come
from v2 with `res_layers = 0`, selected sections come from v3 with their own count. No factor is
refit and no dense reference is read.

Selection ranks each section by its salient-residual mass per residual byte. `sal` in the v2 pack is
`W_sal - (U diag(mid) V diag(pre))_sal` at the `k` columns the pipeline marked as important, so
`rows * k * mean(sal^2)` is the fit error sampled where the activations are largest: the metric the
residual is meant to move, not an unweighted Frobenius norm. The greedy stops at the byte budget.

Usage:

  nq_splice.py --base nq8b/model.nqp --residual nq8b-v3s-r2/model.nqp \
      --cap-bytes 1024500695 --out nq8b-targeted/model.nqp

  # the same bytes as 1.18 b/w, spent as one plane on 16% of the weights
  nq_splice.py ... --keep 1 --rank energy --out nq8b-r1e/model.nqp

`--keep K` drops residual planes `>= K` from the selected sections and adds their contribution
back into the sparse values, so the sparse columns stay exact and plane 0 is unchanged. The v3
fit is sequential in the planes, so this is a byte-level operation, not a refit. `--rank energy`
orders sections by the v3 plane-0 residual energy per byte instead of the v2 salient mass.

`--v4` writes the spliced pack as version 4: a `k`/`v` section whose `V`, `s_pre`, `s_mid` and
`salient_idx` are byte-identical to the same layer's `q` section stores only `U`, `s_post`,
`salient` and its residual, and binds the rest to the head. That is the qkv dedup the rate counts;
the v3 pack is written to a temporary file and rewritten, not kept.
"""

import argparse
import os
import struct

import numpy as np

MAGIC = b"APNQPK01"
KINDS = ["q", "k", "v", "o", "gate", "up", "down"]
HEADER = 32
V2_HEADER = 32
V3_HEADER = 36
V4_HEADER = 40
SHARED_FIELDS = ("v", "pre", "mid", "idx")


def open_pack(path, want_version):
    with open(path, "rb") as f:
        magic, version, header_bytes, layers, kinds, linears, extra = struct.unpack(
            "<8sIIIIII", f.read(HEADER)
        )
        if magic != MAGIC:
            raise SystemExit(f"{path}: bad magic {magic!r}")
        if version != want_version:
            raise SystemExit(f"{path}: version {version}, want {want_version}")
        if header_bytes != HEADER or kinds != len(KINDS) or extra != 0:
            raise SystemExit(f"{path}: header {header_bytes}/{kinds}/{extra} is not the format")
        return dict(path=path, version=version, layers=layers, linears=linears)


def section(f, at, version):
    """Read one section header and locate its payload fields, without reading the payload."""
    f.seek(at)
    raw = f.read(V3_HEADER if version == 3 else V2_HEADER)
    layer, kind, out, inn, rank, u_words, v_words, k = struct.unpack_from("<IIIIIIII", raw)
    res_layers = struct.unpack_from("<I", raw, 32)[0] if version == 3 else 0
    sizes = dict(
        u=out * u_words * 4,
        v=rank * v_words * 4,
        pre=inn * 2,
        mid=rank * 2,
        post=out * 2,
        idx=k * 4,
        sal=out * k * 2,
        res=res_layers * out * v_words * 4,
        s_res=res_layers * out * 2,
    )
    head = V3_HEADER if version == 3 else V2_HEADER
    return dict(layer=layer, kind=kind, out=out, inn=inn, rank=rank, u_words=u_words,
                v_words=v_words, k=k, res_layers=res_layers, sizes=sizes,
                payload=sum(sizes.values()), at=at, head=head,
                sal_at=at + head + sizes["u"] + sizes["v"] + sizes["pre"] + sizes["mid"]
                + sizes["post"] + sizes["idx"],
                s_res_at=at + head + sum(sizes.values()) - sizes["s_res"])


def truncate_payload(raw, r, keep):
    """Drop residual planes `>= keep`, refitting the sparse values to the kept planes.

    `residual_planes` fits the planes in order from `R_0 = W − Ŵ`, so plane 0 does not depend on
    later planes and the kept payload is a prefix of the source's. The sparse values absorb every
    plane at the salient columns (`S = W_sal − (U·V)_sal − Σ_l R_l_sal`), so dropping planes
    `>= keep` adds their contribution back: `S_keep = S_source + Σ_{l ≥ keep} s_l·sign_l`.
    """
    s = r["sizes"]
    at = 0
    parts = []
    for field in ("u", "v", "pre", "mid", "post", "idx", "sal", "res", "s_res"):
        if field in ("res", "s_res"):
            break
        parts.append(raw[at:at + s[field]])
        at += s[field]
    sal_at = at - s["sal"]
    res_at = at
    sal = np.frombuffer(raw[sal_at:sal_at + s["sal"]], "<f2").astype(np.float32)
    sres = np.frombuffer(raw[res_at + s["res"]:res_at + s["res"] + s["s_res"]], "<u2")
    alpha = ((sres.astype(np.uint32) << 16).view(np.float32).astype(np.float64))
    (out, vw, rl) = (r["out"], r["v_words"], r["res_layers"])
    k = s["idx"] // 4
    idx = np.frombuffer(parts[5], "<u4").astype(np.int64)
    words = np.frombuffer(raw[res_at:res_at + s["res"]], "<u4").reshape(rl, out, vw)
    alpha = alpha.reshape(rl, out)
    sal = sal.reshape(out, k)
    for l in range(keep, rl):
        bits = (words[l][:, idx >> 5] >> (idx & 31).astype(np.uint32)) & 1
        sal += np.where(bits == 1, alpha[l][:, None], -alpha[l][:, None])
    return (b"".join(parts[:-1]) + sal.astype("<f2").tobytes()
            + raw[res_at:res_at + keep * out * vw * 4]
            + raw[res_at + s["res"]:res_at + s["res"] + keep * out * 2])


def field_offsets(s):
    """The offset of each payload field from the section's payload start, in file order."""
    out = {}
    at = 0
    for field in ("u", "v", "pre", "mid", "post", "idx", "sal", "res", "s_res"):
        out[field] = at
        at += s["sizes"][field]
    return out


def shared_block(raw, s, offsets):
    """The bytes of `V`, `s_pre`, `s_mid` and `salient_idx` in a payload."""
    return b"".join(raw[offsets[f]:offsets[f] + s["sizes"][f]] for f in SHARED_FIELDS)


def member_payload(raw, s):
    """The payload a version-4 member stores: every field except the shared block."""
    parts = []
    at = 0
    for field in ("u", "v", "pre", "mid", "post", "idx", "sal", "res", "s_res"):
        size = s["sizes"][field]
        if field not in SHARED_FIELDS:
            parts.append(raw[at:at + size])
        at += size
    return b"".join(parts)


def rewrite_v4(src, dst):
    """Rewrite a version-3 pack as version 4, binding each matching `k`/`v` block to its layer's
    `q` section. The membership test is byte equality of `V`, `s_pre`, `s_mid` and `salient_idx`,
    the same test the Rust reader and writer apply."""
    meta = open_pack(src, 3)
    sections = []
    at = HEADER
    with open(src, "rb") as f:
        for _ in range(meta["linears"]):
            s = section(f, at, 3)
            sections.append(s)
            at += s["head"] + s["payload"]
    if at != os.path.getsize(src):
        raise SystemExit(f"{src}: headers walk to {at} B, file is {os.path.getsize(src)} B")
    offsets = {(s["layer"], s["kind"]): field_offsets(s) for s in sections}
    by_layer = {}
    for s in sections:
        by_layer.setdefault(s["layer"], {})[s["kind"]] = s
    members = set()
    with open(src, "rb") as f:
        for layer, group in by_layer.items():
            q = group.get(0)
            if q is None:
                continue
            f.seek(q["at"] + q["head"])
            head = shared_block(f.read(q["payload"]), q, offsets[(layer, 0)])
            for kind in (1, 2):
                s = group.get(kind)
                if s is None:
                    continue
                f.seek(s["at"] + s["head"])
                if shared_block(f.read(s["payload"]), s, offsets[(layer, kind)]) == head:
                    members.add((layer, kind))
    saved = 0
    with open(src, "rb") as f, open(dst, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<IIIIII", 4, HEADER, meta["layers"], len(KINDS),
                              meta["linears"], 0))
        for s in sections:
            f.seek(s["at"] + s["head"])
            raw = f.read(s["payload"])
            if len(raw) != s["payload"]:
                raise SystemExit(f"blk.{s['layer']}.{KINDS[s['kind']]}: short read")
            shared = (s["layer"], s["kind"]) in members
            if shared:
                saved += sum(s["sizes"][field] for field in SHARED_FIELDS)
                raw = member_payload(raw, s)
            out.write(struct.pack("<IIIIIIIIII", s["layer"], s["kind"], s["out"], s["inn"],
                                  s["rank"], s["u_words"], s["v_words"], s["k"],
                                  s["res_layers"], int(shared)))
            out.write(raw)
    want = os.path.getsize(src) + 4 * meta["linears"] - saved
    got = os.path.getsize(dst)
    if got != want:
        raise SystemExit(f"{dst}: wrote {got} B, the accounting says {want} B")
    return dict(members=len(members), saved=saved, size=got,
                payload=got - HEADER - V4_HEADER * meta["linears"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the residual-free v2 pack")
    ap.add_argument("--residual", required=True, help="the uniform two-plane v3 pack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-bytes", type=int, default=1024500695,
                    help="pack cap after the qkv dedup; the spliced pack may exceed it by the dedup")
    ap.add_argument("--no-dedup", action="store_false", dest="dedup",
                    help="do not count the qkv duplicate the format still stores")
    ap.add_argument("--v4", action="store_true",
                    help="write version 4, binding each matching k/v block to its layer's q")
    ap.add_argument("--list", default="", help="write the ranking and the selection here")
    ap.add_argument("--planes", type=int, default=2)
    ap.add_argument("--keep", type=int, default=0,
                    help="residual planes kept per selected section (default: --planes)")
    ap.add_argument("--rank", choices=("sal", "energy"), default="sal",
                    help="section ranker: the v2 salient fit error, or the v3 plane-0 residual energy")
    ap.add_argument("--pick", default="",
                    help="comma list of layer:kind[+kind...], e.g. 33:q+k+o,34:gate+up, "
                         "instead of the greedy or --layers")
    ap.add_argument("--layers", default="",
                    help="comma list of layers: select them instead of the greedy budget")
    ap.add_argument("--kinds", default="q,k,v,o,gate,up,down",
                    help="comma list of kinds for --layers")
    args = ap.parse_args()
    keep = args.keep or args.planes
    if not 0 < keep <= args.planes:
        raise SystemExit(f"--keep {keep}: the source carries {args.planes} planes")

    base = open_pack(args.base, 2)
    res = open_pack(args.residual, 3)
    if (base["layers"], base["linears"]) != (res["layers"], res["linears"]):
        raise SystemExit("the packs cover different sections")

    # Walk both packs' headers.
    sections = []
    at_b = at_r = HEADER
    with open(args.base, "rb") as fb, open(args.residual, "rb") as fr:
        for i in range(base["linears"]):
            b = section(fb, at_b, 2)
            r = section(fr, at_r, 3)
            if (b["layer"], b["kind"], b["out"], b["inn"], b["rank"], b["k"]) != (
                    r["layer"], r["kind"], r["out"], r["inn"], r["rank"], r["k"]):
                raise SystemExit(f"section {i}: the packs disagree on geometry")
            if r["res_layers"] not in (0, args.planes):
                raise SystemExit(
                    f"section {i}: residual {r['res_layers']} planes, want 0 or {args.planes}")
            sections.append((b, r))
            at_b += b["head"] + b["payload"]
            at_r += r["head"] + r["payload"]
    if at_b != os.path.getsize(args.base) or at_r != os.path.getsize(args.residual):
        raise SystemExit(f"headers walk to {at_b}/{at_r} B, files are "
                         f"{os.path.getsize(args.base)}/{os.path.getsize(args.residual)}")

    # The qkv V/pre/mid/idx payload is written three times; the dedup keeps one copy per group.
    dedup = 0
    for layer in range(base["layers"]):
        group = [b for b, _ in sections if b["layer"] == layer and b["kind"] in (0, 1, 2)]
        if len(group) != 3:
            raise SystemExit(f"layer {layer}: {len(group)} qkv sections")
        b = group[0]
        dedup += 2 * (b["sizes"]["v"] + b["sizes"]["pre"] + b["sizes"]["mid"] + b["sizes"]["idx"])
    header_growth = 4 * base["linears"]
    budget = args.cap_bytes - os.path.getsize(args.base) - header_growth
    budget += dedup if args.dedup else 0

    # Read the base `sal` of every section: the fit error at the salient columns.
    rows = []
    with open(args.base, "rb") as f:
        for b, r in sections:
            f.seek(b["sal_at"])
            sal = np.frombuffer(f.read(b["sizes"]["sal"]), "<f2").astype(np.float64)
            mass = float(np.sum(sal ** 2))
            cost = keep * b["out"] * b["v_words"] * 4 + keep * b["out"] * 2
            rows.append(dict(b=b, r=r, mass=mass, cost=cost,
                             weights=b["out"] * b["inn"],
                             rel=float(np.sqrt(np.mean(sal ** 2)))))

    total_mass = sum(x["mass"] for x in rows)
    with open(args.residual, "rb") as f:
        for (_, r), row in zip(sections, rows):
            f.seek(r["s_res_at"])
            a = np.frombuffer(f.read(r["out"] * 2), "<u2")
            a = ((a.astype(np.uint32) << 16).view(np.float32)).astype(np.float64)
            row["energy"] = float(np.sum(a ** 2)) * r["v_words"] * 32
    total_energy = sum(x["energy"] for x in rows)
    total_weights = sum(x["weights"] for x in rows)
    if args.rank == "energy":
        def key(i):
            return rows[i]["energy"] / rows[i]["cost"]
    else:
        def key(i):
            return rows[i]["mass"] / rows[i]["cost"]
    order = sorted(range(len(rows)), key=key, reverse=True)
    if args.pick:
        want = set()
        for item in args.pick.split(","):
            item = item.strip()
            if not item:
                continue
            layer_s, _, kinds_s = item.partition(":")
            for k in kinds_s.split("+"):
                k = k.strip()
                if k not in KINDS:
                    raise SystemExit(f"--pick: unknown kind {k!r} in {item!r}")
                want.add((int(layer_s), k))
        chosen = {i for i, (b, _) in enumerate(sections)
                  if (b["layer"], KINDS[b["kind"]]) in want}
        have = {(sections[i][0]["layer"], KINDS[sections[i][0]["kind"]]) for i in chosen}
        if want - have:
            raise SystemExit(f"--pick: no such sections {sorted(want - have)}")
        spent = sum(rows[i]["cost"] for i in chosen)
    elif args.layers:
        want_layers = {int(x) for x in args.layers.split(",") if x.strip()}
        want_kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
        bad = want_kinds - set(KINDS)
        if bad:
            raise SystemExit(f"--kinds: unknown {sorted(bad)}")
        chosen = {i for i, (b, _) in enumerate(sections)
                  if b["layer"] in want_layers and KINDS[b["kind"]] in want_kinds}
        spent = sum(rows[i]["cost"] for i in chosen)
        if not chosen:
            raise SystemExit("--layers selects no section")
    else:
        chosen = set()
        spent = 0
        for i in order:
            if spent + rows[i]["cost"] <= budget:
                chosen.add(i)
                spent += rows[i]["cost"]
    picked = [rows[i] for i in order if i in chosen]
    mass_share = sum(x["mass"] for x in picked) / total_mass
    energy_share = sum(x["energy"] for x in picked) / total_energy
    cov = sum(x["weights"] for x in picked) / total_weights
    out_bytes = os.path.getsize(args.base) + header_growth + spent
    print(f"dedup {dedup} B  header growth {header_growth} B  budget {budget} B  "
          f"spent {spent} B", flush=True)
    print(f"selected {len(picked)}/{len(rows)} sections  weights {cov:.4%}  "
          f"salient-mass {mass_share:.4%}  energy {energy_share:.4%}  "
          f"lift {mass_share / cov:.2f}x", flush=True)
    print(f"pack {out_bytes} B, rate {out_bytes * 8 / total_weights:.6f} b/w before dedup, "
          f"{(out_bytes - dedup) * 8 / total_weights:.6f} b/w after", flush=True)

    # Write: a v3 file whose unselected sections carry res_layers = 0.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    v3_out = args.out + ".v3" if args.v4 else args.out
    lines = ["layer,kind,out,inn,rank,k,mass_share,energy_share,rel_sal,cost,selected"]
    with open(args.base, "rb") as fb, open(args.residual, "rb") as fr, \
            open(v3_out, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<IIIIII", 3, HEADER, base["layers"], len(KINDS), base["linears"], 0))
        for i, (b, r) in enumerate(sections):
            take_res = i in chosen
            f = fr if take_res else fb
            src = r if take_res else b
            f.seek(src["at"] + src["head"])
            payload = f.read(src["payload"])
            if len(payload) != src["payload"]:
                raise SystemExit(f"blk.{b['layer']}.{KINDS[b['kind']]}: short read")
            if take_res and keep < r["res_layers"]:
                payload = truncate_payload(payload, r, keep)
            out.write(struct.pack("<IIIIIIIII", b["layer"], b["kind"], b["out"], b["inn"],
                                  b["rank"], b["u_words"], b["v_words"], b["k"],
                                  keep if take_res else 0))
            out.write(payload)
            x = rows[i]
            lines.append(f"{b['layer']},{KINDS[b['kind']]},{b['out']},{b['inn']},{b['rank']},"
                         f"{b['k']},{x['mass'] / total_mass:.6f},{x['energy'] / total_energy:.6f},"
                         f"{x['rel']:.6f},{x['cost']},{int(take_res)}")
    got = os.path.getsize(v3_out)
    if got != out_bytes:
        raise SystemExit(f"wrote {got} B, the accounting says {out_bytes}")
    if args.v4:
        r = rewrite_v4(v3_out, args.out)
        os.remove(v3_out)
        print(f"v4 {r['size']} B, {r['payload'] * 8 / total_weights:.6f} b/w, "
              f"{r['members']} members sharing {r['saved']} B", flush=True)
    if args.list:
        with open(args.list, "w") as f:
            f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} {os.path.getsize(args.out)} B", flush=True)


if __name__ == "__main__":
    main()
