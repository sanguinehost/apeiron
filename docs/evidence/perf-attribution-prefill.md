# Performance attribution: prefill, decode, speculation (Qwen3.5-0.8B Q8_0)

Grade: `measured` (i9-14900KF, 32 CPUs, governor `powersave`, quiet gate: host busy < 5% over
1 s before every round). Source: `examples/engine_profile.rs`; tracer `src/engine/trace.rs`.
Receipts: `docs/evidence/bench/*.{json,md}` (identity, per-round samples, traced phases).

## Instrument

| part | what it records |
|---|---|
| `engine::trace` (`APEIRON_TRACE=1`) | per phase: ns, spans, columns, model-derived weight bytes and multiply-adds; off = one relaxed load |
| `pool.wait` overlay | per pool dispatch: the dispatching thread's wait after its own share (straggler tail) |
| `engine_profile` | identity (CPU, governor, kernel, rustc, git, sha256 of `src/`, model, ids), 1 GiB read ceiling at 8/16/32 threads, quiet gate, interleaved `llama-bench` over thread counts, wall + CPU time + MHz per workload, traced run |
| `engine_profile --arm name:ENV=…` | interleaved A/B: each arm is its own process per round, order rotated per round; median of per-round ratios |

Two instrument defects were found and fixed before any number below was taken:

- `llama-bench` inherited the pool's CPU-0 pin (tg 22 instead of 47 tok/s). The child's affinity is
  reset in `run_llama`.
- An idle pool worker spun 5 ms after every 50 ms park (`chart/pool.rs` `worker`), about 2 of 32
  CPUs busy while idle. Host busy fell from 12.3% to 4.9% with the fix.

## Baseline against llama.cpp `0324696b8`, best thread count per round

`1790593468-q35-sliced.md`, 5 rounds, pp 200, tg 64:

| metric | apeiron | llama.cpp | ratio |
|---|---:|---:|---:|
| prefill tok/s | 113.7 | 470.9 (24 threads) | 0.24 |
| decode tok/s | 58.6 | 46.8 (8 threads) | 1.18 |

Earlier comparisons used `llama-bench -t 8` for prefill (≈260 tok/s); llama.cpp's prefill scales
to 24 threads, its decode peaks at 8.

## Where prefill time goes (traced, 8.6 ms/token)

| phase | ms/token | note |
|---|---:|---|
| projections (attn, affine, FFN) | 5.1 | 80–108 GMAC/s at `k = 8` columns |
| affine heads | 1.6–1.7 | per-token recurrence |
| attention core | 1.0 | 6 layers; far above its ≈2.5M multiply-adds |
| affine conv | 0.5 | |
| norms, residuals | 0.26 | |
| `pool.wait` (overlay) | 3.15 | 36% of wall; 187 dispatches/token |

`perf record` (prefill child, 143k samples): 64% of all samples are pool workers spinning; every
worker spins 47–72% of its time. In `q8_0_row_kcols_avx2` the int8 dot (`vpdpbusd`) is ≈3% of the
kernel's samples; the per-block reduction (`vphaddd`), lane extraction and the scalar exact
`i64 × i64 → i128` scale multiply-add dominate.

## A/B results

| change | pp | tg | spec | receipt |
|---|---:|---:|---:|---|
| affine heads split into 4 value-column slices (vs whole) | 1.083 | 1.033 | 1.055 | `1790593686-ab-head-slices` |
| 8 slices (vs whole) | 1.071 | 1.051 | 1.071 | same |
| P-cores only, `APEIRON_ECORES=0` (vs 8P+16E) | 0.976 | 1.006 | – | `1790594041-ab-ecores` |
| 8P+8E (vs 8P+16E) | 1.030 | 0.909 | – | same |

The 16 efficiency cores contribute nothing measurable to either phase.

Slicing is exact: the delta rule is separable by value column, so a slice is a head of value width
`w` (`mixer::tests::sliced_heads_equal_whole_heads`, 2/4/8 slices, pooled and serial; mutation
control shifting a slice's value range: baseline green, mutated red, restored green).

## Why speculation does not pay

At 70% acceptance a round yields 1.7 tokens and needs 1.7× the single-token cost per round to break
even. Traced (`spec` vs `tg` in `1790593468-q35-sliced.md`): one two-column FFN pass costs
0.298 ms against 0.181 ms for one column (1.65×). The per-column work (affine heads, conv, attention,
norms, the per-block scale epilogue) does not amortize. Neither do the MTP block, the draft head or
the rollback (0.48 ms/token). End to end the gain is 3–7%. The kernel microbenchmark's 1.02× for
two columns measured the DRAM stream alone. In the model, the exact scale epilogue makes the
two-column projection compute-bound.

## Open levers, in order of measured weight

1. Projection kernel epilogue: the exact per-(row, block, column) scale product. The same cost
   limits prefill and speculation.
2. Dispatch structure: 187 (prefill) and 308 (decode) dispatches per token, with a 23–36% straggler
   wait. Fuse per-column dispatches (attention, conv) into one per layer.
3. Attention core in prefill: per column, per head; batch the block's queries per head.
