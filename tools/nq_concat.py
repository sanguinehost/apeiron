#!/usr/bin/env python3
"""Concatenate single-layer `APNQPK01` packs into one layer-major pack.

`examples/nq_fit` fits and writes one layer per run when a per-layer rank table is wanted; the
sections are independent and the v2 format is layer-major, so the full pack is the concatenation of
the per-layer section blocks with the global `linears` word patched. No factor is refit and no
section is rewritten: every section header and payload is copied byte for byte.

The tool checks what the reader checks: magic, version, header length, kind count, section order,
and that each section's payload length follows from its own header. It then prints the file bytes,
the stored payload, the replaced weights and both rates (the file rate counts the headers; the
stored rate is `NqPack::bits_per_weight`).

Usage:

  nq_concat.py --out model-100.nqp --layers 36 layers/l0.nqp layers/l1.nqp ...
"""

import argparse
import hashlib
import struct
import sys

MAGIC = b"APNQPK01"
HEADER = 32
V2_SECTION = 32
KINDS = 7


def parse(path):
    b = open(path, "rb").read()
    if len(b) < HEADER or b[:8] != MAGIC:
        sys.exit(f"{path}: not {MAGIC!r}")
    version, hlen, layers, kinds, linears, reserved = struct.unpack_from("<6I", b, 8)
    if version != 2:
        sys.exit(f"{path}: version {version}, want 2")
    if hlen != HEADER:
        sys.exit(f"{path}: header {hlen}, want {HEADER}")
    if kinds != KINDS:
        sys.exit(f"{path}: kinds {kinds}, want {KINDS}")
    out, at = [], HEADER
    for _ in range(linears):
        if at + V2_SECTION > len(b):
            sys.exit(f"{path}: truncated section at {at}")
        layer, kind, o, i, rank, uw, vw, k = struct.unpack_from("<8I", b, at)
        payload = 4 * (o * uw + rank * vw + k) + 2 * (i + rank + o + o * k)
        end = at + V2_SECTION + payload
        if end > len(b):
            sys.exit(f"{path}: section {layer}/{kind} needs {end}, file is {len(b)}")
        out.append((layer, kind, o, i, end - at, b[at:end]))
        at = end
    if at != len(b):
        sys.exit(f"{path}: {len(b) - at} trailing bytes")
    return layers, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="accept a subset of the 252 sections (a campaign check, not a servable pack)",
    )
    ap.add_argument("packs", nargs="+")
    a = ap.parse_args()
    sections, layers = [], None
    for path in a.packs:
        n, sec = parse(path)
        if layers is None:
            layers = n
        elif n != layers:
            sys.exit(f"{path}: layers {n}, first pack {layers}")
        sections.extend(sec)
    if layers != a.layers:
        sys.exit(f"pack layers {layers}, --layers {a.layers}")
    # One section per `(layer, kind)`: the inputs may be layer packs or single-section packs, in any
    # order. A duplicate is a mistake in the input set, not a merge.
    sections.sort(key=lambda s: (s[0], s[1]))
    for prev, cur in zip(sections, sections[1:]):
        if (prev[0], prev[1]) == (cur[0], cur[1]):
            sys.exit(f"duplicate section {cur[0]}/{cur[1]}")
    want = [(l, k) for l in range(a.layers) for k in range(KINDS)]
    got = [(l, k) for l, k, *_ in sections]
    if got != want and not a.allow_partial:
        sys.exit(f"{len(got)} sections, want {len(want)} in layer-major kind order")
    head = MAGIC + struct.pack("<6I", 2, HEADER, layers, KINDS, len(sections), 0)
    out = head + b"".join(raw for *_, raw in sections)
    weights = sum(o * i for _, _, o, i, _, _ in sections)
    payload = sum(size for *_, size, _ in sections)
    stored = payload - V2_SECTION * len(sections)
    with open(a.out, "wb") as f:
        f.write(out)
    h = hashlib.sha256(out).hexdigest()
    print(f"pack {a.out} {len(out)} B, {len(sections)} sections, sha256 {h}")
    print(
        f"stored {stored} B over {weights} weights = {stored * 8 / weights:.9f} b/w; "
        f"file {len(out) * 8 / weights:.9f} b/w"
    )
    if stored * 8 != weights:
        print(f"note: stored bits differ from weights by {stored * 8 - weights}")


if __name__ == "__main__":
    main()
