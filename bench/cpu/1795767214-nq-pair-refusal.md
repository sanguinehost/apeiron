# The fused gate/up pair is refused at verify-pass widths

Criterion: **T1** decode rate; **Q1** unchanged (the emitted stream is the greedy stream).
Grade: `measured` on the box (8 vCPU EPYC Milan, `APEIRON_THREADS=8`). The laptop pair is
**not** in this receipt: `MemAvailable` stayed at 0.91–1.01 GiB against the 2 GiB guard, so the
i5-10210U confirmation is open. Box results are CPU-tier evidence and a timed box result is not a
laptop result.

## Mechanism

`project_gate_up_swiglu_capped` ran the ffn gate and up projections as one column-band dispatch at
every column count. A verify pass folds `k ≤ 8` columns, one column per pool task, so that dispatch
gave each task one column and read **both** linears' `V` and `U` planes once per column, with the
SwiGLU epilogue on the column's own critical path. At `k ≤ NQ_TILE_COLS` the function now returns
`None`; the caller runs gate, up and `swiglu_cols` separately, and each projection gets the pool's
column bands with its own row block. Prefill (`k = 1,827`) keeps the fused pair, where neither cost
applies.

## Numbers

Command, per run, box:

```sh
APEIRON_THREADS=8 target/release/examples/lookup_accept \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp --first 42 --n 128 --ngram 2 --draft 8
```

| arm | runs (ms/token) | mean | vs llama 10.92 |
|---|---|---|---|
| fused pair (`d0052bba`) | 5.940, 6.043, 6.133 | 6.039 | 0.553× |
| pair refused | 5.758, 5.770, 5.852 | 5.793 | 0.531× |

The target is 5.46 ms/token (0.5×); the treated mean is 0.33 ms short. Every run is exact against
the greedy stream: 28 rounds, 102 drafted, 102 accepted.

`perf stat`, lookup pass alone: instructions 55.908e9 → 54.930e9 (−1.8%), cycles 23.052e9 →
22.413e9 (−2.8%), IPC 2.43 → 2.45 (3.649 GHz).

Phase split of the treated lookup decode (`--trace`), ms/token and share:

| phase | ms/token | share |
|---|---|---|
| ffn | 2.200 | 38.2% |
| head | 1.133 | 19.7% |
| attn.proj_in | 0.920 | 16.0% |
| attn.core | 0.637 | 11.1% |
| attn.proj_out | 0.442 | 7.7% |
| pool.wait | 0.299 | 5.2% |
| norm | 0.169 | 2.9% |
| residual | 0.073 | 1.3% |
| embed | 0.006 | 0.1% |

`ffn` is the dominant term and was the target; its `1/(1 − 0.382) = 1.618` ceiling is the pass's
own, not the token's. The token's untraced remainder is 0.051 ms (0.9%).

## Refused: the small-`k` row split

One further body was built and refused. `project_rows_split` gave each participant a band of factor
rows and folded it against every activation column at once through the `C`-column tile
(`code_select_row_c_avx2`), so each factor word was decoded once instead of once per column. It cut
instructions 54.9e9 → 39.0e9 but dropped IPC 2.45 → 1.72; cycles rose and the wall lost. Paired
interleaved A/B on the treated tree (`APEIRON_NQ_COLS=1` = column split, `8` = row split):

| arm | runs (ms/token) | mean |
|---|---|---|
| column split | 5.852, 5.707, 5.840 | 5.800 |
| row split | 6.005, 5.948, 5.958 | 5.970 |

The tile's per-row dependency chain and the `z`/`x` reader quantization repeated in every row task
cost more than the shared decode saved. The equality test held bit for bit at every `k` in `1..=8`,
so the refusal is throughput, not arithmetic. The wide tile's partial-group generality and its test
stay (`78258961`); nothing dispatches a partial group.

## Scope

Covers the 0.6B `nq06-v3` pack and the lookup verify pass on the box. Does not cover the 7B pack,
prefill (unchanged, `k = 1,827` keeps the pair), the laptop, rustc/LLVM code generation, or the
`unsafe` container.
