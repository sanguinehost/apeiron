# nq: prefill dials — query tile neutral, pack geometry tool

Criterion: **T2** (turn latency), prefill half. This is a dial-and-instrument change, not a speed
claim: `APEIRON_QUERY_TILE` replaces the fixed 64-column attention tile so the tile can be swept
without a rebuild, and `examples/nq_stats` prints the pack's per-linear `out × inn × rank` and the
rank-decomposed MACs per token, so a prefill wall is quoted against the arithmetic it ran.

Grades: `measured` for the four tile points on the box (CPU-tier evidence; the timed target is the
laptop, which did not run them); the points span 7.6% and no setting wins outside the unmeasured
spread, so the dial is **neutral** and 64 stays the default. `measured` for the pack geometry.

## Machine, command, binary

Box `ubuntu@139.99.223.161` (8 vCPU EPYC Milan, 32 GiB, `~/bin/adg`), `APEIRON_THREADS=8`.
Model `skeleton-q8head.gguf`, pack `nq06-v3.nqp`. Prefill of **1827 tokens**:

```sh
APEIRON_THREADS=8 APEIRON_QUERY_TILE=<n> target/release/examples/prefill_width \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp --tokens 1827 --width 32
```

| `APEIRON_QUERY_TILE` | wall ms | tok/s |
|---|---|---|
| 32 | 3905 | 467.9 |
| 64 (default) | 3685 | 495.8 |
| 128 | 3701 | 493.7 |
| 256 | 3628 | 503.6 |

The 32-column tile loses 6.0%; 128 and 256 sit within 1.6% of 64, which is inside the run-to-run
spread this harness shows on unchanged code. No change claimed.

## Pack geometry (`nq_stats --pack nq06-v3.nqp`)

```text
linears 196; factor bytes 53808384; bits/weight 0.977441; rank sum 124544
rank-decomposed: 0.4184 GMAC/token = 0.837 GFLOP/token
dense equivalent: 0.881 GFLOP/token
residual: 0 layer-planes, 0.0000 GMAC/token; salient 0.0000 GMAC/token
density 0.950
```

The pack folds 0.4184 GMAC per token against the dense model's 0.4404; at 1827 tokens one prefill
is **764.4 GMAC**. The measured 3.53–3.90 s wall is therefore 196–217 GMAC/s aggregate, 24–27
GMAC/s per thread at width 8.

## What this leaves

The prefill wall is at llama parity and 1.97× above the 0.5× turn budget; a tile dial does not move
it. The next measurement is the thread-scaling and instruction count of the projection kernel
(`code_select_row8_avx2`, 28.93% of the flat prefill profile) against the arithmetic it folds.
