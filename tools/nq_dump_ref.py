#!/usr/bin/env python3
"""Write the source NanoQuant factor's dense reconstruction for each layer linear.

The output is `{out}/l{l}.{kind}-nqref.i64`: an `out`, `inn` little-endian `u64` header and the
`out × inn` Q32.32 words, the layout `tests/learn_nanoquant.rs::read_words` and `cmp_line` read.
`recon_rows` includes the factor's sparse salient term, so this is the source's served
reconstruction, not the rank factor alone.

Usage: nq_dump_ref.py --source qwen3-0-6b-nanoquant.gguf --out DIR [--layers 0,27]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import nq_convert
import nq_gguf

# The four linears the residual table covers: pack kind -> the source's tensor prefix.
KINDS = [("down", "ffn_down"), ("gate", "ffn_gate"), ("up", "ffn_up"), ("o", "attn_output")]
ONE = 1 << 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="")
    args = ap.parse_args()
    layers = (
        [int(x) for x in args.layers.split(",") if x.strip()]
        if args.layers
        else list(range(28))
    )
    src = nq_gguf.parse_gguf(args.source)
    tensors = {t["name"]: t for t in src["tensors"]}
    os.makedirs(args.out, exist_ok=True)
    for l in layers:
        for kind, prefix in KINDS:
            key = f"blk.{l}.{prefix}"
            if key + ".nq_u" not in tensors:
                print(f"skip {key}: no factor", flush=True)
                continue
            f = nq_convert.linear_payload(src, tensors, key)
            rec = nq_convert.recon_rows(f, 0, f["out"])
            words = np.rint(rec.astype(np.float64) * ONE).astype("<i8")
            path = os.path.join(args.out, f"l{l}.{kind}-nqref.i64")
            with open(path, "wb") as out:
                out.write(np.asarray([f["out"], f["inn"]], dtype="<u8").tobytes())
                out.write(words.tobytes())
            print(
                f"{path} {f['out']}x{f['inn']} rank {f['rank']} k {f['k']} "
                f"{os.path.getsize(path)} B",
                flush=True,
            )


if __name__ == "__main__":
    main()
