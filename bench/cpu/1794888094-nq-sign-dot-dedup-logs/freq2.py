import struct
import numpy as np

PACK = '/home/lucas/models/arelath/nq8b/model.nqp'
names = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
f = open(PACK, 'rb')
assert f.read(8) == b'APNQPK01'
_, hb, _, _, n, _ = struct.unpack('<IIIIII', f.read(24))
f.seek(hb)

rows = {}
for s in range(n):
    h = f.read(32)
    layer, kind, out, inn, rank, uw, vw, k = struct.unpack('<8I', h)
    ub, vb = out * uw * 4, rank * vw * 4
    uraw = f.read(ub)
    vraw = f.read(vb)
    f.seek((inn + rank + out) * 2 + k * 4 + out * k * 2, 1)
    if kind == 0 and layer in (0, 17, 35):
        rows[layer] = (out, rank, uw, uraw)

lut = np.array([bin(x).count('1') for x in range(256)], dtype=np.int32)

for layer, (out, rank, uw, uraw) in sorted(rows.items()):
    a = np.frombuffer(uraw, dtype=np.uint32).reshape(out, uw)
    nn = np.empty(out, dtype=np.int32)
    for i in range(out):
        d = np.bitwise_xor(a, a[i]).view(np.uint8)
        pc = lut[d].sum(axis=1)
        pc[i] = 10**9
        nn[i] = pc.min()
    bits = rank
    print(f'layer {layer} q U: {out} x {bits} bits; '
          f'min {nn.min()} p1 {np.percentile(nn,1):.0f} p5 {np.percentile(nn,5):.0f} '
          f'median {np.median(nn):.0f} mean {nn.mean():.0f}; chance {bits/2:.0f}')
    for t in (64, 128, 256, 512):
        print(f'  rows whose nearest neighbour is within {t} bits: '
              f'{(nn <= t).sum()} ({(nn <= t).mean()*100:.2f}%)')
