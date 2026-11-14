#!/usr/bin/env python3
"""Exact Eckart-Young floor for a GGUF Q8_0 linear, at a set of ranks.

Reads only the named tensor, streams rows into a float32 Gram, then eigvalsh.
The floor is the best relative Frobenius error of any rank-r factor.
Usage: spectrum.py <gguf> <tensor> <rank> [rank ...]
"""
import struct, sys
import numpy as np

GGML_Q8_0 = 8
BLOCK = 34  # 2-byte f16 scale + 32 int8

class R:
    def __init__(self, f):
        self.f = f
    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"wanted {n} got {len(b)}")
        return b
    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def i32(self): return struct.unpack("<i", self.take(4))[0]
    def f32(self): return struct.unpack("<f", self.take(4))[0]
    def string(self):
        n = self.u64()
        return self.take(n).decode("utf-8", "replace")

def read_kv(r, t):
    if t == 0: return r.take(1)[0]
    if t == 1: return struct.unpack("<b", r.take(1))[0]
    if t == 2: return struct.unpack("<H", r.take(2))[0]
    if t == 3: return struct.unpack("<h", r.take(2))[0]
    if t == 4: return r.u32()
    if t == 5: return r.i32()
    if t == 6: return r.f32()
    if t == 7: return r.take(1)[0] != 0
    if t == 8: return r.string()
    if t == 9:
        et = r.u32(); n = r.u64()
        return [read_kv(r, et) for _ in range(n)]
    if t == 10: return r.u64()
    if t == 11: return struct.unpack("<q", r.take(8))[0]
    if t == 12: return struct.unpack("<d", r.take(8))[0]
    raise ValueError(f"kv type {t}")

def parse(path, want):
    f = open(path, "rb")
    r = R(f)
    assert r.take(4) == b"GGUF", "not gguf"
    ver = r.u32()
    nt = r.u64(); nkv = r.u64()
    align = 32
    for _ in range(nkv):
        k = r.string(); t = r.u32(); v = read_kv(r, t)
        if k == "general.alignment":
            align = int(v)
    tensors = {}
    for _ in range(nt):
        name = r.string(); nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        tt = r.u32(); off = r.u64()
        tensors[name] = (dims, tt, off)
    data_start = (f.tell() + align - 1) // align * align
    if want not in tensors:
        f.close()
        raise KeyError(f"{want} not in {len(tensors)} tensors, e.g. {list(tensors)[:8]}")
    dims, tt, off = tensors[want]
    return f, data_start + off, dims, tt, ver

def gram(path, name):
    f, at, dims, tt, ver = parse(path, name)
    assert tt == GGML_Q8_0, f"type {tt} not Q8_0"
    inn = dims[0]; out = int(np.prod(dims[1:]))
    assert inn % 32 == 0
    row_bytes = inn // 32 * BLOCK
    small = min(out, inn)
    G = np.zeros((small, small), dtype=np.float32)
    chunk = max(1, min(out, 4_000_000 // inn))
    f.seek(at)
    for r0 in range(0, out, chunk):
        r1 = min(out, r0 + chunk)
        raw = np.frombuffer(f.read((r1 - r0) * row_bytes), dtype=np.uint8)
        blk = raw.reshape(r1 - r0, inn // 32, BLOCK)
        d = blk[:, :, :2].copy().view("<f2").astype(np.float32).reshape(r1 - r0, inn // 32, 1)
        q = blk[:, :, 2:].view(np.int8).astype(np.float32)
        x = (q * d).reshape(r1 - r0, inn)
        if inn <= out:
            G += x.T @ x
        else:
            G[r0:r1, r0:r1] += x @ x.T
    f.close()
    return G, out, inn

def main():
    path, name = sys.argv[1], sys.argv[2]
    ranks = [int(x) for x in sys.argv[3:]]
    G, out, inn = gram(path, name)
    ev = np.linalg.eigvalsh(G)[::-1]
    total = float(ev.sum())
    print(f"{name}: out={out} inn={inn} energy={total:.6g}")
    for r in ranks:
        keep = float(ev[:r].sum())
        rel = ((total - keep) / total) ** 0.5
        print(f"  r={r:5d} ({r/min(out,inn):6.3f} of min)  EY rel={rel:.6f}  sq={rel*rel:.6f}")

if __name__ == "__main__":
    main()
