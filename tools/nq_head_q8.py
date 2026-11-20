#!/usr/bin/env python3
"""Add a `Q8_0` `output.weight` to a GGUF skeleton by quantizing its tied `token_embd.weight`.

The engine prefers `output.weight` for the LM head and falls back to the tied embedding. A tied
`F16` head has no fused row reader (`q8act_plan(F16)` is the `Reference` scalar dequantization), so
decode reads 2 bytes and ~17 cycles per weight per token. A `Q8_0` head takes the fused block reader:
1.0625 bytes per weight (34 per 32) and the packed `vpmaddubsw` row kernel.

The quantization is the reference `Q8_0` one: per 32-weight block, `d = max|x| / 127` stored as
`f16`, `q = round(x / d)` clamped to `[-127, 127]`. The stored image is what the reader decodes, so
the head's logits after this tool are the exact integer projection of that image, not of the `f16`
weights; compare against the original before serving.

Usage:
  nq_head_q8.py --in skeleton.gguf --out skeleton-q8head.gguf
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_gguf

Q8_0 = 8
ROWS_PER_CHUNK = 2048


def encode(src, name, dims, out):
    """Stream `dims` row-major `f16` weights from `src` to `out` as `Q8_0` blocks."""
    n_in, rows = dims
    if n_in % 32:
        raise SystemExit(f"{name}: {n_in} columns is not a multiple of 32")
    row_bytes = n_in * 2
    got = 0
    with open(src, "rb") as f:
        ten = next(t for t in nq_gguf.parse_gguf(src)["tensors"] if t["name"] == name)
        f.seek(ten["abs"])
        while got < rows:
            take = min(ROWS_PER_CHUNK, rows - got)
            raw = f.read(take * row_bytes)
            if len(raw) != take * row_bytes:
                raise SystemExit(f"{name}: short at row {got}")
            x = np.frombuffer(raw, "<f2").astype(np.float32).reshape(take, n_in // 32, 32)
            amax = np.abs(x).max(axis=2)
            d = amax / 127.0
            id = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
            q = np.rint(x * id[..., None]).clip(-127, 127).astype(np.int8)
            d16 = d.astype("<f2")
            block = np.zeros((take, n_in // 32, 34), dtype=np.uint8)
            block[:, :, :2] = d16.view(np.uint8).reshape(take, n_in // 32, 2)
            block[:, :, 2:] = q.view(np.uint8)
            out.write(block.tobytes())
            got += take
    return rows * (n_in // 32) * 34


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    a = ap.parse_args()
    h = nq_gguf.parse_gguf(a.src)
    if any(t["name"] == "output.weight" for t in h["tensors"]):
        raise SystemExit(f"{a.src}: already carries output.weight")
    emb = next((t for t in h["tensors"] if t["name"] == "token_embd.weight"), None)
    if emb is None or emb["type"] != 1:
        raise SystemExit(f"{a.src}: token_embd.weight is not F16")
    n_in, vocab = emb["dims"]
    nbytes = vocab * (n_in // 32) * 34

    entries = [nq_gguf.frozen_entry(h, t) for t in h["tensors"]]

    def write_head(out, src=a.src, name=emb["name"], dims=emb["dims"], nbytes=nbytes):
        wrote = encode(src, name, dims, out)
        if wrote != nbytes:
            raise SystemExit(f"output.weight: wrote {wrote} B, want {nbytes}")

    entries.append(
        {
            "name": "output.weight",
            "dims": list(emb["dims"]),
            "type": Q8_0,
            "nbytes": nbytes,
            "write": write_head,
        }
    )
    total = nq_gguf.build_gguf_stream(a.dst, h["meta"], entries)
    print(f"{a.dst}: {total} B, {len(entries)} tensors, output.weight Q8_0 {nbytes} B")


if __name__ == "__main__":
    sys.exit(main())
