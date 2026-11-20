#!/usr/bin/env python3
"""Build a compact Qwen3-8B skeleton whose output head is the stock Q6_K tensor.

The source skeleton stores a Q8_0 output head at data offset 0.  Replacing it with the
stock Q6_K head (150,720,512 B smaller) and shifting every later tensor down by that
difference keeps the file layout and drops the dead Q8_0 payload.
"""

import os
import struct
import sys

SRC = "/home/lucas/models/arelath/nq8b/skeleton.gguf"
HEAD = "/home/lucas/models/qwen3-8b/head-q6k/output.weight.bin"
OUT = "/home/lucas/models/arelath/nq8b/skeleton-q6k.gguf"

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def parse_header(f):
    f.seek(0)
    magic = f.read(4)
    if magic != b"GGUF":
        raise SystemExit(f"bad magic {magic!r}")
    version = struct.unpack("<I", f.read(4))[0]
    nt = struct.unpack("<Q", f.read(8))[0]
    nkv = struct.unpack("<Q", f.read(8))[0]

    def rdstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def val(t):
        if t == 8:
            return rdstr()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [val(et) for _ in range(n)]
        b = f.read(SZ[t])
        return struct.unpack(FMT[t], b)[0]

    keys = {}
    for _ in range(nkv):
        k = rdstr()
        t = struct.unpack("<I", f.read(4))[0]
        keys[k] = val(t)

    tensors = []
    for _ in range(nt):
        name_pos = f.tell()
        name = rdstr()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        type_pos = f.tell()
        tt = struct.unpack("<I", f.read(4))[0]
        off_pos = f.tell()
        off = struct.unpack("<Q", f.read(8))[0]
        n = 1
        for d in dims:
            n *= d
        tensors.append(dict(name=name, dims=dims, type=tt, off=off, n=n, name_pos=name_pos, type_pos=type_pos, off_pos=off_pos))

    info_end = f.tell()
    align = keys.get("general.alignment", 32)
    data_start = (info_end + align - 1) // align * align
    return dict(version=version, nt=nt, keys=keys, tensors=tensors, info_end=info_end, align=align, data_start=data_start)


def main():
    with open(SRC, "rb") as f:
        h = parse_header(f)
        size = os.path.getsize(SRC)

    out_t = next(t for t in h["tensors"] if t["name"] == "output.weight")
    tok_t = next(t for t in h["tensors"] if t["name"] == "token_embd.weight")
    if out_t["type"] != 8:
        raise SystemExit(f"source output type is {out_t['type']}, expected Q8_0 (8)")
    if tok_t["off"] <= out_t["off"]:
        raise SystemExit("token_embd is not after output.weight")

    q8_bytes = out_t["n"] // 32 * 34
    q6_bytes = os.path.getsize(HEAD)
    shift = q8_bytes - q6_bytes
    if shift <= 0:
        raise SystemExit("Q6_K head is not smaller than Q8_0")
    if tok_t["off"] - out_t["off"] != q8_bytes + 16384:
        print(f"note: source gap is {tok_t['off'] - out_t['off'] - q8_bytes}, expected 16384")

    # Header bytes, with the output type/offset and every later tensor offset patched.
    with open(SRC, "rb") as f:
        header = bytearray(f.read(h["data_start"]))
        f.seek(h["data_start"] + tok_t["off"])
        tail_start = f.tell()
        tail_len = size - tail_start

    struct.pack_into("<I", header, out_t["type_pos"], 14)  # GGML_TYPE_Q6_K
    struct.pack_into("<Q", header, out_t["off_pos"], 0)
    for t in h["tensors"]:
        if t["name"] == "output.weight":
            continue
        new_off = t["off"] - shift
        if new_off < 0:
            raise SystemExit(f"negative offset for {t['name']}")
        struct.pack_into("<Q", header, t["off_pos"], new_off)

    new_tail = tok_t["off"] - shift
    new_data = q6_bytes + 16384 + tail_len
    new_size = h["data_start"] + new_data
    print(f"source {size} B, data_start {h['data_start']}, Q8 head {q8_bytes}, Q6 head {q6_bytes}, shift {shift}")
    print(f"output {OUT}: {new_size} B, token offset {new_tail}, tail {tail_len} B")

    with open(OUT, "wb") as out, open(HEAD, "rb") as head, open(SRC, "rb") as src:
        out.write(header)
        while True:
            b = head.read(8 << 20)
            if not b:
                break
            out.write(b)
        out.write(b"\0" * 16384)
        src.seek(tail_start)
        left = tail_len
        while left:
            b = src.read(min(8 << 20, left))
            if not b:
                raise SystemExit("short tail read")
            out.write(b)
            left -= len(b)

    if os.path.getsize(OUT) != new_size:
        raise SystemExit("output size mismatch")

    with open(OUT, "rb") as f:
        v = parse_header(f)
    o = next(t for t in v["tensors"] if t["name"] == "output.weight")
    tok = next(t for t in v["tensors"] if t["name"] == "token_embd.weight")
    if (o["type"], o["off"]) != (14, 0):
        raise SystemExit(f"bad output descriptor {o['type']} {o['off']}")
    if tok["off"] != new_tail:
        raise SystemExit(f"bad token offset {tok['off']} != {new_tail}")
    print(f"verified: output Q6_K off 0, token off {tok['off']}, version {v['version']}, tensors {v['nt']}")


if __name__ == "__main__":
    main()
