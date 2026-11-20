#!/usr/bin/env python3
"""Upgrade a version-1 `APNQPK01` pack to version 3 by inserting the fields version 1 lacks.

Version 1 writes a 28-byte section header `(layer, kind, out, inn, rank, u_words, v_words)` and the
payload `U, V, s_pre, s_post`. Version 3 writes a 36-byte header that appends the salient count and
the residual layer count, and a payload that inserts `s_mid` between `s_pre` and `s_post`. Every
version-1 section this tool has seen carries no salient term and no residual, so the upgrade is
`k = 0`, `res_layers = 0`, `s_mid = 1.0` (bfloat16 `0x3F80`) per rank, and every other byte copied
verbatim. It is an exact upgrade, not a refit: the served matrix of an upgraded section equals the
version-1 section's own reconstruction.

The scale words are bfloat16, the encoding `src/format/nq.rs` decodes (`bf16_to_fixed`) and
`tools/nq_pack_factors.py` writes. This was checked on the 0.6B pack: decoding them as float16 gives
a 1.4e3 relative error against the served dense, decoding them as bfloat16 gives 0.

Usage:
  nq_upgrade_v1.py --in model.nqp --out model-v3.nqp

Refuses a pack whose version is not 1, whose header length is not 32, whose sections do not consume
the buffer exactly, or whose header count disagrees with the sections read.
"""

import argparse
import struct
import sys

MAGIC = b"APNQPK01"
VERSION_IN = 1
VERSION_OUT = 3
HEADER_BYTES = 32
HDR_IN = 28
HDR_OUT = 36
BF16_ONE = 0x3F80
KINDS = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    data = memoryview(open(args.src, "rb").read())
    if len(data) < HEADER_BYTES:
        raise SystemExit(f"{args.src}: {len(data)} B is shorter than the {HEADER_BYTES}-byte header")
    magic, version, hlen, layers, kinds, linears, reserved = struct.unpack_from(
        "<8sIIIIII", data, 0
    )
    if magic != MAGIC:
        raise SystemExit(f"{args.src}: magic {magic!r}, want {MAGIC!r}")
    if version != VERSION_IN:
        raise SystemExit(f"{args.src}: version {version}, want {VERSION_IN}")
    if hlen != HEADER_BYTES:
        raise SystemExit(f"{args.src}: header {hlen} B, want {HEADER_BYTES}")
    if kinds != KINDS:
        raise SystemExit(f"{args.src}: {kinds} kinds, want {KINDS}")
    if linears > layers * kinds:
        raise SystemExit(f"{args.src}: {linears} sections for {layers} layers × {KINDS} kinds")

    out = bytearray()
    out += struct.pack("<8sIIIIII", MAGIC, VERSION_OUT, HEADER_BYTES, layers, kinds, linears, 0)
    at = HEADER_BYTES
    for i in range(linears):
        if at + HDR_IN > len(data):
            raise SystemExit(f"section {i}: header at {at} runs past {len(data)} B")
        layer, kind, rows, inn, rank, uw, vw = struct.unpack_from("<7I", data, at)
        at += HDR_IN
        if layer >= layers or kind >= kinds or rows == 0 or inn == 0 or rank == 0:
            raise SystemExit(
                f"section {i}: layer {layer} kind {kind} out {rows} in {inn} rank {rank} refuses"
            )
        if uw != (rank + 31) // 32 or vw != (inn + 31) // 32:
            raise SystemExit(f"section {i}: strides {uw}/{vw} disagree with rank {rank} in {inn}")
        u_bytes = 4 * rows * uw
        v_bytes = 4 * rank * vw
        pre_bytes = 2 * inn
        post_bytes = 2 * rows
        end = at + u_bytes + v_bytes + pre_bytes + post_bytes
        if end > len(data):
            raise SystemExit(f"section {i}: payload to {end} runs past {len(data)} B")
        out += struct.pack("<9I", layer, kind, rows, inn, rank, uw, vw, 0, 0)
        out += data[at : at + u_bytes]
        at += u_bytes
        out += data[at : at + v_bytes]
        at += v_bytes
        out += data[at : at + pre_bytes]
        at += pre_bytes
        out += struct.pack("<H", BF16_ONE) * rank
        out += data[at : at + post_bytes]
        at += post_bytes
    if at != len(data):
        raise SystemExit(f"{len(data) - at} B after the last section, want 0")
    with open(args.dst, "wb") as f:
        f.write(out)
    print(
        f"{args.dst}: {len(data)} -> {len(out)} B, {linears} sections, "
        f"{layers} layers × {kinds} kinds, version {VERSION_OUT}"
    )


if __name__ == "__main__":
    sys.exit(main())
