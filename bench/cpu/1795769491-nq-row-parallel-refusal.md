# NQ decode: the row-parallel split is refused at verify-pass widths

Criterion: **T1** decode rate (0.5× llama.cpp). Grade: `measured` on the OVH box for the
walls, instruction and cycle counts and the symbol profile; `tested` for the kernel equality
tests. The laptop pair is open (`MemAvailable` below the 2 GiB guard). The timed claim names the
box; it is CPU-tier evidence, not laptop evidence.

## Result

Two row-parallel bodies were built and paired against the committed column-band body. Both cut
the projection's instruction count and both lose the wall, because at decode widths a linear is
only 5–33 µs of work across the pool: a second dispatch costs more than the columns it saves.

| draft 8, `lookup_accept`, ms/token | control `493bcffc` | row-parallel `V`+`U` | `U`-only row split |
|---|---|---|---|
| run 1 | 5.802 | 7.113 | 6.768 |
| run 2 | 5.850 | 7.121 | 6.815 |
| run 3 | 5.852 | 7.180 | 6.801 |
| mean | 5.835 | 7.138 (+22.3%) | 6.795 (+16.4%) |

The `U`-only arm keeps the column-band `V` stage (so `t_build` stays parallel) and splits only
the `U` rows. Its control batch ran 5.975 / 5.957 / 5.896 (mean 5.943), so the paired delta is
**+0.85 ms/token** — the cost of one extra dispatch per linear, `196 × 4.4 µs` per token.

Instructions fall and cycles rise, the signature of a latency loss:

| draft 8 | instructions e9 | cycles e9 | IPC |
|---|---|---|---|
| control | 139.469 / 139.616 | 76.833 / 78.003 | 1.82 / 1.79 |
| row-parallel `V`+`U` | 124.757 / 124.310 | 82.916 / 79.699 | 1.50 / 1.56 |

At the widths where one group of eight covers the pass the row split does win: draft 3 (`k = 4`)
measured 7.944 / 7.817 / 7.854 control against 7.437 / 7.722 / 7.686 row-parallel, a 3.3% win on
the mean. Draft 7 (`k = 8`) still loses: 5.848 / 5.868 / 5.926 against 7.129 / 7.180 / 7.215.

## Machine, binaries and command

Machine: OVH box `ubuntu@139.99.223.161` (8 vCPU EPYC Milan, 32 GiB), `APEIRON_THREADS=8`,
`~/bin/adg`. CPU-tier evidence; the timed target is the laptop (i5-10210U), which did not run
this measurement.

Model `skeleton-q8head.gguf`; pack `nq06-v3.nqp` (196 linears, 0.977441 b/w).

| arm | binary | sha256 |
|---|---|---|
| control (`493bcffc`) | `/tmp/lookup_treat` | `4f3d7e1ec864469ff796c7e6417095abcc6d423bb7a888ed5ca3396f51b9b4cb` |
| row-parallel `V`+`U` | `/tmp/lookup_rows` | `16f24444c061a07d71af7427756f60bef40b0ffc9e0f5b7cb24760654e350ffd` |
| `U`-only row split | `/tmp/lookup_urows` | `afe46070e4e1c087502a329e90a632990727a7ecb2308034a39f765606ca93fc` |

```sh
APEIRON_THREADS=8 /tmp/lookup_{treat,rows,urows} \
  --model skeleton-q8head.gguf --pack nq06-v3.nqp --first 42 --n 128 --ngram 2 --draft 8
```

Both arms are exact against the greedy stream: rounds 28, accepted 102, drafted 102, `exact`.

## Mechanism

The row-parallel body gives each participant a contiguous run of factor rows and folds every
activation column against it, one `StageCols::dot_row` per group of eight columns. The
column-band body instead gives each participant one column and reads the whole factor plane per
column. With `k` at most the pool width the band body leaves `width − k` participants idle, and
with `k ≤ 8` its band is one column, so the eight-column tile never engages.

That is why instructions fall 11%: one eight-column call replaces eight one-column passes over
the same factor row. The wall rises anyway:

- The dispatching thread builds `t` (`k · inn` `mul_q32`, the scalar pre-scale) and quantizes it
  before the `V` dispatch, because every row's dot needs every `t` word. In the band body that
  work is inside each task, split by column. The `U`-only arm removes that cost and still loses,
  so the barrier itself is the floor.
- A dispatch that publishes to seven workers and joins them costs about **4.4 µs** here
  (`0.85 ms/token ÷ 196 linears`). A linear's whole band-body span at `k = 5` is 15–35 µs, and
  the widest recorded phase, `ffn`, is 2.220 ms/token for three linears per layer.

The profile agrees: on the control, `code_select_rows_avx2` (the blocked per-column reader) is
15.74% of user cycles; on the row-parallel arm it falls to 8.83% and the wide row tile
`code_select_row_c_avx2::<7>` appears at 2.77%, while `Pool::with_classes` (the worker loop,
which inlines the tasks) rises from 46.98% to 53.17% — the extra epochs and their joins.

## Phase split and ceiling

Draft 8, control binary, `--trace` (ms/token and share of the 5.806 ms/token wall):

| phase | ms/token | share |
|---|---:|---:|
| ffn | 2.220 | 38.2% |
| head | 1.136 | 19.6% |
| attn.proj_in | 0.918 | 15.8% |
| attn.core | 0.653 | 11.2% |
| attn.proj_out | 0.444 | 7.6% |
| pass | 4.545 | 78.3% |
| pool.wait | 0.382 | 6.6% |

The dominant leaf is `ffn`, so its pipeline ceiling is `1/(1 − 0.382) = 1.62×`; the whole
projection pass is 78.3%, ceiling `4.6×`. The decode target is `5.46 ms/token` (half of llama's
`10.92`); `5.835` is `1.07×` short, inside `ffn`'s ceiling. This change does not collect it: a
split that adds a barrier per linear cannot land inside a phase whose per-linear span is shorter
than the barrier's cost.

## Scope

- Box only; no laptop timed row. CPU-tier evidence.
- The equality tests pass for both bodies (`the_pooled_block_is_bit_equal_at_every_block`,
  `the_dense_epilogue_rounds_the_scaled_dot`, extended to `k = 1..=8` at `tile = NQ_TILE_COLS`).
  The full lib suite was not rerun on the refused arms; the tree is reverted to `493bcffc`.
- No code is committed from this receipt. The refusal is the artifact: a future row split must
  either amortise several linears per dispatch or remove the `V`/`U` barrier, not add one.
- Rust/LLVM codegen, the kernel ABI and the `unsafe` container are outside the claim.

## Files

- `src/engine/kern/nq.rs`: reverted to `493bcffc` (no row-parallel body remains).
