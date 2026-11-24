# One factor read per column group: 1.16x on the packed prefill, 3.9x behind llama.cpp cold

Criterion: **T2** (turn latency), partial. Grade: `measured`. Machine: ThinkPad L13 Yoga 20R6,
i5-10210U (4 cores / 8 threads, AVX2, no AVX-VNNI), 6.6 GiB, Ubuntu 24.04, `APEIRON_THREADS=4`,
governor `powersave`. State: idle for the paired runs (load 0.85, 3.9 GHz, 54 °C); the traced split
ran at load 2.5, 86 °C. Both are named in the tables.

Model: `~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf`; pack `nq06-v3.nqp`
(196 linears, 28 layers, ranks 480/640/736, no salient indices, no residual layer, 0.977441 b/w,
53,815,472 B, 440,401,920 dense weights). Tokenizer `qwen3-0.6b-base-e8/tokenizer.json`.

## What changed

`proj_cols` already pooled the resident `Q8_0`/`Q4_K`/`Q6_K` tensors across a block's columns. A
packed linear did not: each column called `nq_proj` again, and every call re-read and re-decoded all
196 linears' `V` and `U` words. At 1,827 positions that is 98.3 GB of pack reads per cold prefill.

`NqLinear::project_pooled`, `project_q8_pooled` and `project_ternary_pooled` keep each column's own
pre-scaled words, quantization, reader and accumulation order, and give the group one `V` dispatch
and one `U` dispatch: a task reads each factor row once and folds it against every column it owns.
`proj_cols` takes the pooled entry for `xs.len() > 1`; a refusal reruns the group through the
per-column entry, which keeps that entry's own reader fallback per column. `k = 1` still runs the
single-column path, which is also the oracle.

Tests, `cargo test --lib --locked -j 1 engine::kern::nq::tests` (14 passed):

- `the_pooled_readers_equal_the_per_column_readers` — exact, Q8 and ternary pooled entries equal
  the per-column entries bit for bit at `k = 1, 2, 3, 7, 32, 33, 256` on a fixture carrying a
  sparse correction and a dense residual.
- `the_pooled_reader_skips_the_x_stage_without_a_residual` — the served pack's geometry.
- `the_nq_prefill_equals_the_per_column_prefill` (`engine::step::prefill_tests`) — the whole
  executor on a packed fixture: one column per block and 59 columns per block produce the same
  prefill token, the same next-token logits word for word and the same cache length, under
  `KernelMode::Exact` and `KernelMode::Q8Act`. A `+2^20` mutation of one pooled output word turns
  it red; `+1` (the smallest nontrivial Q32.32 perturbation) is absorbed by the forward's rounding,
  which is why the mutation uses `2^20`.
- `the_wide_accumulator_agrees_when_the_bound_leaves_i64` covers the pooled entries too.

## Same-binary A/B: 1.16x where the projection is 51% of the turn

`APEIRON_PREFILL_COLS=1` makes every block one column, so `proj_cols` takes the per-column entry for
the whole prefill; `256` takes the pooled entry. Interleaved, order reversed each round, same
binary, same pack, `--len 229 --differ 1 --decode 4 --mb 440`. Log
`/tmp/opencode/nq-pooled-ab2.log`.

| round | 1 column/block | 256 columns/block | ratio |
|---|---:|---:|---:|
| 1 | 3.969 s | 3.297 s | 1.204x |
| 2 | 4.160 s | 3.717 s | 1.119x |
| 3 | 4.408 s | 3.872 s | 1.138x |
| 4 | 5.134 s | 4.324 s | 1.187x |
| median of runs | 4.284 s | 3.795 s | 1.129x |

The within-round ratio is the statistic (each pair shares a thermal state): mean **1.162x**, range
1.119–1.204. The first A/B at the same length, always running the pooled side second, gave 1.08x
(medians 4.151 s against 3.845 s, `/tmp/opencode/nq-pooled-ab.log`); the order matters, so the
reversed-order set is the one quoted.

At 1,827 the same A/B is **refused as evidence**: load 6.7 and 96 °C (read from `/proc/loadavg` and
`/sys/class/thermal/thermal_zone0/temp` at the end of the set) inverted the order in round 1
(one-column 85.094 s against pooled 98.275 s) and round 2 finished at 108.679 against 109.631 s
(`/tmp/opencode/nq-pooled-ab1827.log`). The traced split below is the 1,827 figure.

## Why 1.16x and not 229x

The per-column path's reads were already served from cache. Each linear's factor plane is tens of
KiB, and a block of 256 columns reads it 256 times back to back, so the repeated reads hit L2/L3.
The old cold read 98.3 GB in 65.96 s = 1.49 GB/s, far below the box's ~13 GB/s: the bytes were not
the bottleneck. What pooling removes is the per-column weight fetch and dispatch overhead on a path
whose cost is the sign dot, and the sign dot is still run once per column. Pooling cannot remove
arithmetic that is per (column, word).

The ceiling follows from the split: projections hold 51.4% of the cold turn at 1,827, so even a free
projection path caps the cold prefill at 2.06x there. The 1.16x is measured at 229 positions, where
the traced split puts `ffn` at 52.1%; `ffn` is one leaf that mixes the pooled gate/up/down with
swiglu and the norms, so the projection's own share is not decomposable from the trace and the
1.16x is not extrapolated to 1,827.

## Traced split, 1,827 positions (final binary)

`prefix_solve --len 1827 --differ 205 --decode 16 --mb 440 --phases`, wall = prefill + decode:

```
cold   hit     0  prefill   56.761 s (  32.19 tok/s)  decode 16 in  1.076 s (14.87 tok/s)  first 315
  norm            361.776 ms    0.6%  calls 1425
  residual        315.462 ms    0.5%  calls 1400
  attn.proj_in   8281.423 ms   14.3%  calls 700
  attn.core     26852.529 ms   46.4%  calls 700
  attn.proj_out  3822.627 ms    6.6%  calls 700
  ffn           17643.355 ms   30.5%  calls 700
  pass          57335.092 ms   99.1%  calls 25
  pool.wait      1270.647 ms    2.2%  calls 12869
shared hit  1616  prefill   10.575 s ( 172.77 tok/s)  decode 16 in  1.081 s (14.81 tok/s)  first 8630
  attn.proj_in   1074.170 ms    9.2%  calls 504
  attn.core      7441.912 ms   63.9%  calls 504
  attn.proj_out    501.362 ms    4.3%  calls 504
  ffn            2271.102 ms   19.5%  calls 504
  pass          11403.065 ms   97.8%  calls 18
  pool.wait       219.305 ms    1.9%  calls 9138
```

`pass` counts the 16 decode passes as well as the prefill blocks, so its share is of prefill +
decode and the leaves sum to 98.9% cold, 96.9% tail. `pool.wait` is an overlay.

The 211-token tail is 53.1 ms/position and its dominant term is `attn.core` at 63.9%, not a
projection. The cold turn's dominant term is also `attn.core` (46.4%). Pooling the projection is
spent; the next change targets attention or the sign dot itself, not the weight fetch.

## Against llama.cpp, three alternating rounds

Engine (`prefix_solve --len 1827 --differ 205 --decode 16 --mb 440`), llama.cpp
`049326a` (`llama-completion --prompt-cache`, 1,838-token prompt, 187-token delta, `-c 2048 -t 4
-ngl 0 --no-warmup --no-display-prompt -n 16 --temp 0 --seed 1`), alternated within each round.
Log `/tmp/opencode/nq-pooled-vs-llama.log`.

| turn | engine | llama.cpp | ratio |
|---|---:|---:|---:|
| cold prefill, per token | 60.911 s / 1,827 = **33.34 ms** | 15.836 s / 1,838 = **8.62 ms** | **3.87x** |
| 211-token tail, per token | 11.535 s / 211 = **54.7 ms** | 2.023 s / 187 = **10.82 ms** | **5.05x** |
| cold 16-token turn | 62.00 s | 17.43 s | **3.56x** |
| tail 16-token turn | 12.74 s | 3.50 s | **3.64x** |
| repeat 16-token turn | 1.40 s | 1.49 s | engine 1.07x |
| decode | 14.67 tok/s | 10.01 tok/s | engine 1.47x |

Runs: engine cold 53.725 / 61.384 / 60.911 s, tail 11.426 / 11.668 / 11.535 s, decode 15.27 / 14.04
/ 14.67 tok/s. llama.cpp cold prompt eval 20,206.87 / 15,836.42 / 15,711.27 ms, delta 2,023.31 /
2,004.02 / 2,130.23 ms, decode 8.77 / 10.01 / 10.28 tok/s. The turns are the per-round sums, median
over the three: cold engine 54.773 / 62.524 / 62.001 s against llama 22.031 / 17.434 / 17.268 s;
tail engine 12.744 / 12.795 / 12.673 s against llama 3.504 / 3.417 / 3.516 s; repeat engine
1.347 / 1.402 / 1.398 s against llama 1.491 / 1.523 / 1.455 s.

The committed baseline (`1795490586-nq06-vs-llamacpp.md`) measured cold 4.25x and tail 4.88x in a
contended state. The cold ratio moved to 3.87x here; the tail did not move, because the tail is
attention-bound and pooling does not touch attention. Decode is 1.47x faster than llama.cpp on the
same turn because the pack is 2.25x smaller than the f16 GGUF (482,672,896 B skeleton + 53,815,472 B
pack against 1,198,182,592 B).

## The machine ceiling

The cold prefill reads the 53.8 MB pack once per block: 8 blocks = 0.43 GB in 56.761 s, 7.6 MB/s,
two orders under the ~13 GB/s this box has measured (`1795490586-nq06-vs-llamacpp.md`). The packed
projection is not byte-bound; the sign dot issues once per column.

Attention is byte-bound. `attn_core_f32_strided` reads each column's causal prefix, so a cold
1,827-position turn reads `229,376 B/position × Σ_{c=0}^{1826}(c+1)` = **383.0 GB** of KV.
`attn.core` takes 26.85 s, **14.3 GB/s**, at the box's measured bandwidth. A projection-only change
is capped at 2.06x by the split; the tail's ceiling is lower still, because `attn.core` holds 63.9%
of it.

## What this does not cover

- Attention: 46.4% of the cold turn, 63.9% of the tail. `attn_core_f32_strided` reads each
  `(column, kv_head)`'s causal prefix, so a column's whole prefix is streamed per column; llama.cpp
  tiles queries and streams each K/V row once per tile. That is the next change, and it is a
  different order of magnitude from this one.
- The sign dot inside every pooled task is still per column. Pooling removed the repeated factor
  read, not the arithmetic.
- The pooled entry checks the group's column widths before the arithmetic, so a group carrying both
  a wrong width and an overflow reports the width where the per-column loop would report the earlier
  column's overflow. `proj_cols` passes one width per part, so the step path cannot reach the
  difference.
- `proof/regen.sh --check` did not run: the pinned Aeneas/Charon toolchain is not on this box.
  `kern/mod.rs` is included by `#[path]` in `proof/extract/lib.rs` but `nq.rs` has no extraction
  root (`proof/README.md`), so the new functions do not translate; the invariant is the Rust
  equality tests above, which is the grade stated.
- Full `cargo test --lib --all-features` on this tree: 673 passed, 45 failed, all 45 Vulkan
  (`vkCmdWriteTimestamp2 not found via vkGetDeviceProcAddr`) on a box with no usable device; the 14
  NQ tests, the two prefill tests and every prefix test pass.
- `cargo clippy --workspace --all-features --all-targets --locked -j 1 -- -D warnings` and
  `cargo fmt --all -- --check` pass. The example's pre-existing `cast_precision_loss` /
  `cast_possible_truncation` lints (in the committed print code) are allowed at the module with a
  reason.

## Commands

```sh
# A/B, one column per block against 256
APEIRON_THREADS=4 APEIRON_PREFILL_COLS=$cols target/release/examples/prefix_solve \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len 229 --differ 1 --decode 4 --mb 440

# split and the llama.cpp turns: as 1795490586-nq06-vs-llamacpp.md, with --phases
```
