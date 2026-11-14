#!/usr/bin/env python3
"""Replace every non-linear tensor of a Q8_0 image with the same tensor from a base image.

The NanoQuant skeleton and a dense Q8_0 reference can carry the same architecture but different
quantisations of the embedding, head and norms (different quantiser, different imatrix). A bake
that writes the served linears into the dense reference then differs from the served pack by those
tensors alone. This tool grafts the skeleton's tensors into the baked image so the linear fit is
the only variable.

    nq_graft.py --image nq8b-r0.gguf --base skeleton.gguf --out nq8b-r0-g.gguf

Both images are streamed; the output is never resident.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_gguf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="baked Q8_0 image")
    ap.add_argument("--base", required=True, help="image carrying the non-linear tensors to keep")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img = nq_gguf.parse_gguf(args.image)
    base = nq_gguf.parse_gguf(args.base)
    base_ten = {t["name"]: t for t in base["tensors"]}
    img_names = set(t["name"] for t in img["tensors"])

    entries = []
    grafted = 0
    for ten in img["tensors"]:
        src_ten = base_ten.get(ten["name"])
        if src_ten is None:
            entries.append(nq_gguf.frozen_entry(img, ten))
            continue
        if src_ten["dims"] != ten["dims"] or src_ten["type"] != ten["type"]:
            raise SystemExit(f"{ten['name']}: base {src_ten['dims']}/{src_ten['type']} "
                             f"!= image {ten['dims']}/{ten['type']}")
        entries.append(nq_gguf.frozen_entry(base, src_ten))
        grafted += 1

    missing = set(base_ten) - img_names
    total = nq_gguf.build_gguf_stream(args.out, img["meta"], entries)
    print(f"wrote {args.out}: {total} B, {grafted} tensors grafted, "
          f"{len(img['tensors']) - grafted} kept, {len(missing)} base-only skipped")


if __name__ == "__main__":
    main()
