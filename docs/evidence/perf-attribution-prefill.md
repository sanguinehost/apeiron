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

## Earlier baseline against llama.cpp `0324696b8`, best thread count per round

`1790593468-q35-sliced.md`, 5 rounds, pp 200, tg 64:

| metric | apeiron | llama.cpp | ratio |
|---|---:|---:|---:|
| prefill tok/s | 113.7 | 470.9 (24 threads) | 0.24 |
| decode tok/s | 58.6 | 46.8 (8 threads) | 1.18 |

Earlier comparisons used `llama-bench -t 8` for prefill (≈260 tok/s); llama.cpp's prefill scales
to 24 threads, its decode peaks at 8.

## Earlier prefill attribution (traced, 8.6 ms/token)

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

## Earlier A/B results

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

## Earlier speculation attribution

At 70% acceptance a round yields 1.7 tokens and needs 1.7× the single-token cost per round to break
even. Traced (`spec` vs `tg` in `1790593468-q35-sliced.md`): one two-column FFN pass costs
0.298 ms against 0.181 ms for one column (1.65×). The per-column work (affine heads, conv, attention,
norms, the per-block scale epilogue) does not amortize. Neither do the MTP block, the draft head or
the rollback (0.48 ms/token). End to end the gain is 3–7%. The kernel microbenchmark's 1.02× for
two columns measured the DRAM stream alone. In the model, the exact scale epilogue makes the
two-column projection compute-bound.

## Earlier priorities

1. Projection kernel epilogue: the exact per-(row, block, column) scale product. The same cost
   limits prefill and speculation.
2. Dispatch structure: 187 (prefill) and 308 (decode) dispatches per token, with a 23–36% straggler
   wait. Fuse per-column dispatches (attention, conv) into one per layer.
3. Attention core in prefill: per column, per head; batch the block's queries per head.


## Batched source baseline and core budget

Source `39298a1c858a11397b7a9e3d638c5a0b796b4a47`; source digest
`96c75f9773af6cb9d0f0ef2536cc7c0c8ccb86020759fe614c214a706300c8de`.
The same model, pp200/tg64, five repetitions and quiet gate are used. The completed integration
checks are recorded in `inference-integration.md`.

| metric | apeiron median | llama.cpp best-thread median | median per-round ratio |
|---|---:|---:|---:|
| prefill tokens/s | 224.99 | 523.54 | 0.43 |
| greedy tokens/s | 62.71 | 49.90 | 1.27 |
| speculative tokens/s, one-token prefill included | 37.95 | not compared | not compared |

Receipt: [`1790607104-astra-batched-baseline.md`](bench/1790607104-astra-batched-baseline.md).
Speculative acceptance is 24%; 39 restore calls occur in the traced run. The current traced
prefill costs 4.33 ms/token: affine heads 1.323 ms, FFN 1.125 ms, convolution 0.519 ms, affine input
projection 0.514 ms and attention core 0.355 ms. Greedy decode costs 15.91 ms/token: the output-head region
4.100 ms and FFN 4.095 ms dominate. This is the measured `engine_profile` workload; it is not a
claim about every public generation API or every prompt.

A separate rotated A/B changes only core and slice settings:

| configuration | prefill tokens/s | greedy tokens/s | greedy ratio to its baseline |
|---|---:|---:|---:|
| default pool | 271.38 | 69.15 | 1.000 |
| 8 performance cores | 230.91 | 67.56 | 0.979 |
| 4 performance cores | 137.92 | 47.84 | 0.693 |
| 4 performance cores, 8 head slices | 133.99 | 47.97 | 0.708 |

Receipt: [`1790607289-astra-core-budget.md`](bench/1790607289-astra-core-budget.md).
Each arm has five samples; every round passed the quiet gate. Compare ratios within a receipt;
the standalone and interleaved-with-llama baselines differ. Core restriction on this i9 is not
measurement on a different low-end machine. The subsequent experiments below retain the original arithmetic and bounds.


## Scheduling experiments rejected

Each receipt contains five quiet rotated rounds at pp200/tg64, the same Qwen3.5-0.8B Q8_0 model
and prompt fixture. These are same-binary schedule comparisons, not fresh llama.cpp comparisons.
The two SwiGLU experiments also measure speculation with 32,768 draft rows.

| candidate / baseline | median paired prefill ratio | four-core prefill ratio | decision | receipt |
|---|---:|---:|---|---|
| affine column traversal / row traversal | 1.000292 | 1.002071 | remove candidate; retain scalar boundary tests | [traversal](bench/1790608521-astra-write-traversal.md) |
| flat-buffer SwiGLU / per-column dispatch | 0.987 | 0.988 | remove candidate | [flat buffer](bench/1790610276-astra-swiglu-batch.md) |
| owned whole-column SwiGLU / per-column dispatch | 1.027322 | 1.001618 | insufficient consistent benefit; remove candidate | [owned columns](bench/1790611132-astra-swiglu-owned.md) |

The owned-column candidate's prefill ratios are 1.060560, 1.031395, 0.974909, 0.993520 and
1.027322. Speculation loses four of five rounds under both core settings (median paired ratios
0.991890 default and 0.988280 four-core). Its median default-pool prefill is 272.67 tokens/s
against 267.27; four-core prefill is 140.14 against 139.51. The measured greedy rates are 67.96
against 68.86 default, and 49.25 against 49.97 four-core. The single-column decode body is
unchanged, so these decode differences do not establish a scheduling benefit. This final
comparison takes 180.88 seconds, exit 0; all 20 samples are complete, quiet, finite and positive.

The final tree restores `chart/ffn.rs` and `step/cols.rs` to `608e726` exactly and has no
`APEIRON_BATCH_SWIGLU` switch or multi-column helper. Three scheduling/traversal experiments
do not establish the required improvement. The next bounded question is the measured contribution
of the exact projection-scale epilogue, separated from dot accumulation with an instrument control.
The 60-token/s low-end target, faster-than-llama prefill and approximately double decode remain open.
