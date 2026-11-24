# The 0.6B prefill cache: cold, warm and a 205-token delta (211-position tail)

Criterion: **T2** (turn latency). Grade: `measured`. Machine: ThinkPad L13 Yoga 20R6, i5-10210U
(4 cores / 8 threads, AVX2, no AVX-VNNI), 6.6 GiB, Ubuntu 24.04, `APEIRON_THREADS=4`. Load average
6–7 during the tabulated runs; the later repeats (11:35–11:47) fell to 4.1 tok/s decode and are
discarded — decode below 14 tok/s is the signature of the page cache no longer holding the model,
not a property of the path.

Model: `~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf` (Q8_0 head, 482 MB).
Pack: `~/models/nq-scratch/nq06-v3.nqp` (196 linears, 0.977441 b/w). Tokenizer:
`~/models/qwen3-0.6b-base-e8/tokenizer.json`. Three runs per length.

## Command

```sh
APEIRON_THREADS=4 target/release/examples/prefix_solve \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len 1827 --differ 205 --decode 16 --mb 900
```

`--len` cycles the prompt's tokens to 1,827 positions; `--differ 205` replaces the last 205 tokens
for the third turn. The binary is `19d24bf4`: the batched tail is in it, the snapshot charge fix
(`e3450821`) is not — the `cache ... 838139904 bytes stored` lines below are that overcharge
(1,827 × 458,752 B); the timing path is unchanged, and the corrected charge of 424,574,976 B
(capacity 1,851 × 229,376 B) fits the same 900 MiB budget. The example now warms the mapped weights
before the cold turn and charges with `estimate_snapshot_bytes`; the tabulated set predates both,
and the cold row is the steady-state prefill it reproduces (28.92/28.05/26.26 against the published
29.44, spread 28.01–31.95, `bench/cpu/1795170568-nq06-context-split.md`).

## 1,827 positions

| run | cold | warm (hit 1,824) | shared (hit 1,616) |
|---|---:|---:|---:|
| 1 | 63.166 s (28.92 tok/s) | 0.268 s | 11.765 s (155.29 tok/s) |
| 2 | 65.138 s (28.05 t/s) | 0.287 s | 12.454 s (146.70 t/s) |
| 3 | 69.577 s (26.26 t/s) | 0.416 s | 13.227 s (138.13 t/s) |
| mean | 65.96 s (27.7 t/s) | 0.324 s (203.8×) | 12.48 s (146.4 t/s, 5.29×) |

Decode, 16 tokens after the turn: cold 14.53/14.30/8.93 tok/s, warm 14.46/14.31/10.87, shared
10.81/11.03/13.53 — the cache holds KV, not weights, so it does not move decode.

Raw:

```
cold   hit     0  prefill   63.166 s (  28.92 tok/s)  decode 16 in  1.101 s (14.53 tok/s)  first 315
warm   hit  1824  prefill    0.268 s (6826.31 tok/s)  decode 16 in  1.106 s (14.46 tok/s)  first 315
shared hit  1616  prefill   11.765 s ( 155.29 tok/s)  decode 16 in  1.480 s (10.81 tok/s)  first 8630
cache 1 entries, 838139904 bytes stored
cold   hit     0  prefill   65.138 s (  28.05 tok/s)  decode 16 in  1.119 s (14.30 tok/s)  first 315
warm   hit  1824  prefill    0.287 s (6370.51 tok/s)  decode 16 in  1.118 s (14.31 tok/s)  first 315
shared hit  1616  prefill   12.454 s ( 146.70 tok/s)  decode 16 in  1.451 s (11.03 tok/s)  first 8630
cache 1 entries, 838139904 bytes stored
cold   hit     0  prefill   69.577 s (  26.26 tok/s)  decode 16 in  1.792 s ( 8.93 tok/s)  first 315
warm   hit  1824  prefill    0.416 s (4386.83 tok/s)  decode 16 in  1.472 s (10.87 tok/s)  first 315
shared hit  1616  prefill   13.227 s ( 138.13 tok/s)  decode 16 in  1.182 s (13.53 tok/s)  first 8630
cache 1 entries, 838139904 bytes stored
```

The shared turn's 211-token tail (`1,827 − 1,616`) costs 59.2 ms/position, not the cold 36.1:
those 211 queries attend to the 1,616 cached keys. The hit is 88.5% of the positions and 81.1% of
the wall. A 16-token turn is 67.2 s cold and 13.7 s shared, 4.9×.

## 512 positions

Same model and pack, `--len 512 --differ 205 --decode 16 --mb 320`:

| run | cold | warm (hit 496) | shared (hit 304) |
|---|---:|---:|---:|
| 1 | 13.560 s (37.76 t/s) | 0.466 s | 5.226 s (97.97 t/s) |
| 2 | 13.408 s (38.19 t/s) | 0.488 s | 5.501 s (93.08 t/s) |
| 3 | 12.041 s (42.52 t/s) | 0.541 s | 11.901 s (43.02 t/s) |

The third shared run is an outlier (2.2× the other two, decode inside its range); the first two
give a 2.4× ratio. The 208-token tail costs 25.8 ms/position against 25.4 cold — at a 304-token
prefix the tail's attention is small, so the tail is priced like the cold average.

Raw (transcribed from the 10:44 runs, same binary and pack):

```
cold   hit     0  prefill   13.560 s (  37.76 tok/s)  decode 16 in  0.704 s (22.72 tok/s)  first 659
warm   hit   496  prefill    0.466 s (1098.00 tok/s)  decode 16 in  0.729 s (21.93 tok/s)  first 659
shared hit   304  prefill    5.226 s (  97.97 tok/s)  decode 16 in  0.807 s (19.82 tok/s)  first 1
cold   hit     0  prefill   13.408 s (  38.19 tok/s)  decode 16 in  0.841 s (19.02 tok/s)  first 659
warm   hit   496  prefill    0.488 s (1048.69 tok/s)  decode 16 in  0.769 s (20.81 tok/s)  first 659
shared hit   304  prefill    5.501 s (  93.08 tok/s)  decode 16 in  0.847 s (18.90 tok/s)  first 1
cold   hit     0  prefill   12.041 s (  42.52 tok/s)  decode 16 in  0.739 s (21.66 tok/s)  first 659
warm   hit   496  prefill    0.541 s ( 946.10 tok/s)  decode 16 in  0.762 s (21.00 tok/s)  first 659
shared hit   304  prefill   11.901 s (  43.02 tok/s)  decode 16 in  0.753 s (21.26 tok/s)  first 1
```

## The charge, after the fix

The later 11:35 run with `--mb 440` printed the corrected charge and is the raw line behind the
note's 405 MiB figure:

```
kv            28 layers, 8 kv heads x 128, 57344 values/position, 229376 B/position charged
len 1827 differ 205 decode 16 cache 440 MiB, 424574976 B needed
cache 1 entries, 424574976 bytes stored
```

That run's timings are discarded (cold 87.8 s, decode 8.2 tok/s against 21 clean); only the charge
line is used.

## What this does not cover

- T2 itself: 1,000-token incremental turn at 200k in ≤ 60 s. This is a 0.6B mechanism measurement
  at 1,827 positions.
- The 8B: no run. Its 2k snapshot is 580 MiB of F32 KV at capacity 2,064, and the linear 8.3×
  projection is an upper bound (the tail's attention holds the 0.6B at 5.3×).
- The hit rate is a workload assumption; these are exact-prefix repeats.
- The 90% figure is the boundary hit (1,616/1,827), not a measured session trace.
