import struct, sys
import numpy as np

PACK = '/home/lucas/models/arelath/nq8b/model.nqp'
names = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
f = open(PACK, 'rb')
assert f.read(8) == b'APNQPK01'
version, hb, layers, kinds, n, _ = struct.unpack('<IIIIII', f.read(24))
f.seek(hb)

want = {}
for s in range(n):
    h = f.read(32)
    layer, kind, out, inn, rank, uw, vw, k = struct.unpack('<8I', h)
    ub, vb = out * uw * 4, rank * vw * 4
    uraw = f.read(ub)
    vraw = f.read(vb)
    rest = (inn + rank + out) * 2 + k * 4 + out * k * 2
    f.seek(rest, 1)
    if layer == 0:
        want[names[kind]] = (out, inn, rank, uw, vw, uraw, vraw)

def popcount(a):
    return np.unpackbits(a.view(np.uint8)).reshape(a.shape[0], -1).sum(axis=1)

def hamming_stats(raw, rows, words, label, queries=256, seed=0):
    a = np.frombuffer(raw, dtype=np.uint32).reshape(rows, words)
    rng = np.random.default_rng(seed)
    q = rng.choice(rows, size=min(queries, rows), replace=False)
    mins = []
    for i in q:
        d = np.bitwise_xor(a, a[i]).view(np.uint8)
        # popcount per row through a 256-entry table
        pc = np.zeros(d.shape[0], dtype=np.int32)
        lut = np.array([bin(x).count('1') for x in range(256)], dtype=np.int32)
        pc = lut[d].sum(axis=1)
        pc[i] = 10**9
        mins.append(int(pc.min()))
    mins = np.array(mins)
    bits = words * 32
    print(f'{label}: {rows} rows x {bits} bits; nearest-neighbour Hamming over {len(q)} queries')
    print(f'  min {mins.min()}  p5 {np.percentile(mins,5):.0f}  median {np.median(mins):.0f}  '
          f'mean {mins.mean():.1f}  random-expectation {bits/2:.0f}  '
          f'min/expected {mins.min()/(bits/2):.3f}')

def gf2_rank(raw, rows, words, label):
    a = np.frombuffer(raw, dtype=np.uint32).reshape(rows, words)
    basis = {}
    for i in range(rows):
        x = int.from_bytes(a[i].tobytes(), 'little')
        while x:
            p = x.bit_length() - 1
            if p in basis:
                x ^= basis[p]
            else:
                basis[p] = x
                break
    print(f'{label}: GF(2) rank {len(basis)} of {rows} rows '
          f'({len(basis)/rows:.4f}); deficiency {rows-len(basis)}')

for kind in ['v', 'q']:
    out, inn, rank, uw, vw, uraw, vraw = want[kind]
    print(f'\n== layer 0, kind {kind}: out {out} inn {inn} rank {rank}')
    hamming_stats(vraw, rank, vw, 'V')
    hamming_stats(uraw, out, uw, 'U')
    gf2_rank(vraw, rank, vw, 'V')
    gf2_rank(uraw, out, uw, 'U')
