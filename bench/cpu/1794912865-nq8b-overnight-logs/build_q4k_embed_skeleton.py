#!/usr/bin/env python3
"""Build a Qwen3-8B skeleton with the stock Q4_K_M token embedding and Q6_K head.

The source skeleton's Q8_0 head and Q8_0 token embedding are replaced by the stock
Q4_K_M file's Q6_K head and Q4_K token embedding.  All other tensors are copied
unchanged and their offsets are rewritten for the smaller payloads.
"""

import os
import struct

SRC = "/home/lucas/models/arelath/nq8b/skeleton.gguf"
HEAD = "/home/lucas/models/qwen3-8b/head-q6k/output.weight.bin"
TOKEN = "/home/lucas/models/qwen3-8b/head-q6k/token_embd_q4k.bin"
OUT = "/home/lucas/models/arelath/nq8b/skeleton-q6k-q4kembed.gguf"

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
Q8_BYTES = 661_225_472
Q6_BYTES = 510_504_960
Q4K_TOKEN_BYTES = 350_060_544
OUT_NORM_BYTES = 16_384
GAP = Q8_BYTES - Q6_BYTES  # 150,720,512


def parse_header(f):
    f.seek(0)
    if f.read(4) != b"GGUF":
        raise SystemExit("bad magic")
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
        return struct.unpack(FMT[t], f.read(SZ[t]))[0]

    keys = {}
    for _ in range(nkv):
        k = rdstr()
        t = struct.unpack("<I", f.read(4))[0]
        keys[k] = val(t)

    tensors = []
    for _ in range(nt):
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
        tensors.append(dict(name=name, dims=dims, type=tt, off=off, n=n, type_pos=type_pos, off_pos=off_pos))

    info_end = f.tell()
    align = keys.get("general.alignment", 32)
    data_start = (info_end + align - 1) // align * align
    return dict(version=version, nt=nt, tensors=tensors, data_start=data_start)


def main():
    with open(SRC, "rb") as f:
        h = parse_header(f)
        size = os.path.getsize(SRC)
    by_name = {t["name"]: t for t in h["tensors"]}
    out_t = by_name["output.weight"]
    norm_t = by_name["output_norm.weight"]
    tok_t = by_name["token_embd.weight"]
    if (out_t["type"], tok_t["type"]) != (8, 8):
        raise SystemExit("source head/token are not Q8_0")
    if norm_t["off"] != out_t["off"] + Q8_BYTES:
        raise SystemExit("output_norm is not immediately after the head")
    if tok_t["off"] != norm_t["off"] + OUT_NORM_BYTES:
        raise SystemExit("token is not immediately after output_norm")
    for t in h["tensors"]:
        if t["name"] in ("output.weight", "output_norm.weight", "token_embd.weight"):
            continue
        if t["off"] <= tok_t["off"]:
            raise SystemExit(f"unexpected tensor before token: {t['name']} at {t['off']}")

    new_norm = Q6_BYTES
    new_tok = new_norm + OUT_NORM_BYTES
    tail_shift = GAP + (Q8_BYTES - Q4K_TOKEN_BYTES)
    tail_start = h["data_start"] + tok_t["off"] + Q8_BYTES
    tail_len = size - tail_start
    new_size = h["data_start"] + new_tok + Q4K_TOKEN_BYTES + tail_len

    with open(SRC, "rb") as f:
        header = bytearray(f.read(h["data_start"]))
        f.seek(h["data_start"] + norm_t["off"])
        norm_bytes = f.read(OUT_NORM_BYTES)

    struct.pack_into("<I", header, out_t["type_pos"], 14)  # Q6_K
    struct.pack_into("<Q", header, out_t["off_pos"], 0)
    struct.pack_into("<Q", header, norm_t["off_pos"], new_norm)
    struct.pack_into("<I", header, tok_t["type_pos"], 12)  # Q4_K
    struct.pack_into("<Q", header, tok_t["off_pos"], new_tok)
    for t in h["tensors"]:
        if t["name"] in ("output.weight", "output_norm.weight", "token_embd.weight"):
            continue
        new_off = t["off"] - tail_shift
        if new_off < 0:
            raise SystemExit(f"negative offset for {t['name']}")
        struct.pack_into("<Q", header, t["off_pos"], new_off)

    print(f"source {size} B, data_start {h['data_start']}, tail {tail_len} B")
    print(f"new {new_size} B, norm {new_norm}, token {new_tok}, tail start {new_tok + Q4K_TOKEN_BYTES}")

    with open(OUT, "wb") as out, open(HEAD, "rb") as head, open(TOKEN, "rb") as token, open(SRC, "rb") as src:
        out.write(header)
        for path, fh, total in ((HEAD, head, Q6_BYTES), (None, None, 0)):
            if path is None:
                break
            left = total
            while left:
                b = fh.read(min(8 << 20, left))
                if not b:
                    raise SystemExit("short head read")
                out.write(b)
                left -= len(b)
        out.write(norm_bytes)
        left = Q4K_TOKEN_BYTES
        while left:
            b = token.read(min(8 << 20, left))
            if not b:
                raise SystemExit("short token read")
            out.write(b)
            left -= len(b)
        src.seek(tail_start)
        left = tail_len
        while left:
            b = src.read(min(8 << 20, left))
            if not b:
                raise SystemExit("short tail read")
            out.write(b)
            left -= len(b)

    if os.path.getsize(OUT) != new_size:
        raise SystemExit("size mismatch")
    with open(OUT, "rb") as f:
        v = parse_header(f)
    vb = {t["name"]: t for t in v["tensors"]}
    if (vb["output.weight"]["type"], vb["output.weight"]["off"]) != (14, 0):
        raise SystemExit("bad output descriptor")
    if (vb["output_norm.weight"]["off"], vb["token_embd.weight"]["type"], vb["token_embd.weight"]["off"]) != (new_norm, 12, new_tok):
        raise SystemExit("bad norm/token descriptor")
    print(f"verified: head Q6_K@0, norm {new_norm}, token Q4_K@{new_tok}, {v['nt']} tensors")


if __name__ == "__main__":
    main()
