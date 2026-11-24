# Tiled attention: each K/V row read once per 32 columns, the tail 1.67x

Criterion: **T2** (turn latency), partial. Grade: `measured` for the timing, `tested` for the
equality. Machine: ThinkPad L13 Yoga 20R6, i5-10210U (4 cores / 8 threads, AVX2, no AVX-VNNI),
6.6 GiB, Ubuntu 24.04, `APEIRON_THREADS=4`, governor `powersave`. State: the box carried another
workspace lane's `cargo clippy --workspace` and `cargo test` throughout
(`/home/lucas/oc-tmp/lanes/w77-compact-clear`), load average 2.9-7.5 at the start/end markers in
`/tmp/opencode/nq-attn-*.log`, 89 °C (`/sys/class/thermal/thermal_zone0/temp`), `/proc/cpuinfo`
reporting 3000.043 MHz under the `powersave` governor. That state inflates every absolute number on
both sides; the paired ratios and the same-window llama.cpp comparison are the evidence, and the
pre-change clean set is quoted separately, not projected through the loaded ratio.

Model: `~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf`; pack `nq06-v3.nqp`
(196 linears, 28 layers, 0.977441 b/w). llama.cpp `049326a` on
`qwen3-0.6b-base-f16/Qwen3-0.6B-Base.f16.gguf` (1,198,182,592 B).

## What changed

`attn.core` held 46.4% of the cold 1,827-position turn and 63.9% of the 211-token tail, reading
`229,376 B/position × Σ_c (c+1)` = 1,669,878 rows × 229,376 B = 383.03 GB of F32 KV per cold
prefill. The F32 block read issued
one task per (column, KV head) and streamed each column's whole causal prefix, so column `c+1`
re-read every row column `c` had just read.

`attn_core_f32_tile` (`src/engine/attn/f32_core.rs`) inverts the loop: a task owns 32 consecutive
columns and one KV head, loads each K and V row once per 256-position pass, and folds it against
every query head of every column in the tile. Each `(column, head)` runs `attn_core_f32_strided`'s
recurrence in the same order — the same 8-lane score chain, running maximum, `exp`, `mul_add`
denominator and per-lane FMA accumulator — so its words equal the per-column call's bit for bit.
`KvState::attn_head_group_tile_f32` is the carrier arm; `softmax_core_block` dispatches one task
per (fold tile, KV head) on the F32 carrier, with the same fold permutation at tile granularity
and a gather back to column order. `k = 1`, `APEIRON_EXACT_SCAN`, `q32` and `i8` keep their
existing paths. `APEIRON_ATTN_TILE=0` selects the per-column dispatch for a same-binary A/B.

The tile kernel's query-head dimension is not incidental: a KV head serves
`group = n_head / n_head_kv = 2` query heads, and the first version assumed one. The engine test
below caught it (`Kv(BadGeometry)` at the first block), and the kernel now takes `query_heads` and
folds the whole group in one pass.

The cold read falls from 383.03 GB to 12.55 GB (`Σ_tiles (last_tile_position + 1) × 229,376 B` =
54,723 rows × 229,376 B), 30.5x fewer bytes. The arithmetic is unchanged: `1,669,878 row visits ×
16 query heads × 128 head_dim × 2 (score and value) × 28 layers` = 191.5 G MACs per cold prefill,
383 GFLOP counted as two operations per MAC.

## Equality tests

- `test_attn_core_query_tile_equals_the_per_query_core`
  (`cargo test --lib --locked -j 1 engine::attn::f32_core::tests::test_attn_core_query_tile`):
  the AVX2 body and the scalar twin equal `attn_core_f32_strided` word for word at `query_heads`
  1/2/4, `head_dim` 8/16/64/128, `stride` = hd and hd+8, first tile positions 0/1/7/255/256/257/600
  and tile sizes 1/2/3/8/33. 4.66 s.
- `the_tile_group_read_equals_the_per_column_group_read`
  (`cargo test --lib --locked -j 1 the_tile_group_read_equals`): a `KvCarrier::F32` `KvState` with
  `group = 2`, every word of `attn_head_group_tile_f32` equals the per-column `attn_head_group`
  at first positions 0/1/7/8/9/31 and tile sizes 1/2/3/8/9, plus first 250 and 255 with tile size
  50, which crosses `KEY_CHUNK = 256` inside one tile pass.
- `the_attention_tile_task_equals_the_per_column_task`
  (`cargo test --lib --locked -j 1 the_attention_tile_task_equals`): the stage-3 task pair on a
  `group = 2`, `n_kv = 2` cache, 35 columns (two tiles, the second partial), `gated = false` and
  `true`. Every gathered tile word equals the per-column task's word at the same `(column, KV
  head)`, and the gated output differs from the ungated one, so the gated leg is not the ungated
  path under another name.
- `the_nq_prefill_equals_the_per_column_prefill`
  (`cargo test --lib --locked -j 1 engine::step::prefill_tests`): the whole executor on a packed
  fixture, 59 tokens, one column per block against 59, under `KernelMode::Exact` and `Q8Act`: all
  59 hidden-state rows, the prefill token, the next-token logits and the cache length are equal.
  The state comparison is the load-bearing one: the token and the next-token logits read only the
  KV rows, which attention never writes, and a zeroed tile output stays green under those two
  alone. Zeroing the tile output turns the state comparison red; restoring it turns it green.

## Same-binary A/B, 1,827 positions, 3 alternating rounds

`APEIRON_ATTN_TILE=$tile`, `--len 1827 --differ 205 --decode 16 --mb 440`, order reversed each
round; `/tmp/opencode/nq-attn-tile-ab.log`.

| round | tiled cold | untiled cold | ratio | tiled tail | untiled tail | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 72.469 s | 101.570 s | 1.402x | 11.245 s | 18.962 s | 1.686x |
| 2 | 88.602 s | 102.458 s | 1.156x | 13.045 s | 21.223 s | 1.627x |
| 3 | 82.625 s | 97.936 s | 1.185x | 11.395 s | 19.207 s | 1.686x |
| median of runs | 82.625 s | 101.570 s | | 11.395 s | 19.207 s | |

Within-round mean paired ratio: cold **1.248x** (1.156-1.402), tail **1.666x** (1.627-1.686). The
tail ratio is the same in both orders; the cold spread follows the thermal drift inside each pair.

## Phase split, tiled against untiled, decode 0

`prefix_solve ... --decode 0 --phases`, two pairs, order reversed; `/tmp/opencode/nq-attn-phase.log`.

| pair | tiled cold attn.core | untiled cold attn.core | tiled tail attn.core | untiled tail attn.core |
|---|---:|---:|---:|---:|
| 1 (tiled first) | 28.204 s | 38.984 s | 5.674 s | 10.694 s |
| 2 (untiled first) | 33.516 s | 52.295 s | 6.147 s | 13.319 s |

Tiled shares: cold `ffn` 37.4/37.6%, `attn.core` 35.1/34.3%, `attn.proj_in` 17.8/18.2%. Tail
`attn.core` 48.7/48.4%, `ffn` 29.4/29.5%. Untiled shares: cold `attn.core` 42.4/45.2%; tail
`attn.core` 62.4/63.4%.

The mechanism is in the bytes: untiled 383.03 GB / 38.98 s = 9.8 GB/s (read-bound, near the box's
measured ~13 GB/s); tiled 12.55 GB / 28.20 s = 0.45 GB/s. The tiled attention is arithmetic-bound
now, at 383 GFLOP / 28.20 s = 13.6 GFLOP/s. The cold turn's dominant term is `ffn` (37.4%), with
`attn.core` one point behind; the tail's dominant term is still `attn.core` at 48.4%.

## Against llama.cpp, same state, 3 alternating rounds

Engine tiled (`--len 1827 --differ 205 --decode 16 --mb 440`), llama.cpp
`llama-completion --prompt-cache` on the f16 GGUF (`-c 2048 -t 4 -ngl 0 --no-warmup
--no-display-prompt -n 16 --temp 0 --seed 1`, 1,838-token prompt, 187-token delta), alternated
within each round; `/tmp/opencode/nq-attn-vs-llama.log`. llama's delta reports `eval time` over 15
runs because the 16-token generation counts the replayed last prompt token as one run; the engine
tail turn decodes 16 tokens. The two per-token figures are both `wall / tokens prefilled`, so the
count difference does not enter the ratio.

| turn | engine tiled | llama.cpp | ratio |
|---|---:|---:|---:|
| cold prefill, per token | 85.729 s / 1,827 = **46.93 ms** | 21.478 s / 1,838 = **11.69 ms** | **4.01x** |
| 211-token tail, per token | 12.487 s / 211 = **59.18 ms** | 2.947 s / 187 = **15.76 ms** | **3.76x** |
| cold 16-token turn | 87.37 s | 23.39 s | **3.73x** |
| tail 16-token turn | 14.34 s | 4.72 s | **3.04x** |
| decode | 10.03 tok/s | 8.24 tok/s | engine 1.22x |

Runs: engine cold 82.033 / 85.729 / 86.759 s, tail 12.224 / 12.851 / 12.487 s, decode 10.09 /
9.76 / 10.03 tok/s. llama cold prompt eval 21,478.49 / 21,260.48 / 22,273.50 ms, delta 2,947.20 /
3,230.03 / 2,936.77 ms, decode 8.36 / 8.24 / 8.08 tok/s. Medians are quoted.

The untiled path in the preceding A/B set (same loaded window, its own legs, 101.570 s cold and
19.207 s tail) is **4.76x** and **5.78x** behind those llama.cpp runs. Tiling moves the tail from 5.78x to 3.76x and the
cold from 4.76x to 4.01x.

The pre-change clean set (`1795498875-nq06-pooled-projection.md`: load 0.85, 54 °C) is 33.34 ms
cold and 54.7 ms tail against llama's 8.62 and 10.82, 3.87x and 5.05x. A clean-window projection
from this loaded ratio is not asserted: the engine tail moved 1.08x between the two windows
(54.7 -> 59.18 ms) while llama.cpp moved 1.46x (10.82 -> 15.76 ms), so the two sides do not share a
single thermal factor.

## What this does not cover

- The clean, unloaded absolute set: every number above was taken at 89 °C with another lane's
  `rustc` on the box. The paired ratios and the alternated llama.cpp legs share that state; the
  clean absolute row is `model`, derived from the pre-change set.
- The tiled kernel is arithmetic-bound, and its next terms are the single-accumulator score chain
  (16 dependent FMAs at hd 128) and the two scalar `exp_f32_scalar` calls per (key, query). Neither
  is changed here. A vector `exp` over the tile's queries would be word-identical only if each
  lane runs the same polynomial and rounding, which is a separate change with its own equality test.
- `attn_core_f32_tile` is not extracted: `f32_core` has no extraction root in `proof/README.md`,
  and the AVX2 body is `tested` equal to the scalar twin, not proved.
- The `unsafe` AVX2 body's safety argument is the feature check plus the geometry the body
  re-checks; rustc, LLVM and the kernel ABI are outside the claim.
- `proof/regen.sh --check` still cannot run on this box (pinned Aeneas/Charon absent).
- Gates: `cargo fmt --all -- --check`, `python3 tools/check-claims.py`,
  `python3 tools/book-check.py` pass. `cargo clippy --workspace --all-features --all-targets
  --locked -j 1 -- -D warnings` is the queued gate; the full lib suite's 45 failures are the known
  Vulkan `vkCmdWriteTimestamp2` device queries.

## Commands

```sh
# A/B (order reversed each round; APEIRON_ATTN_TILE=0 is the per-column dispatch)
APEIRON_THREADS=4 APEIRON_ATTN_TILE=$tile target/release/examples/prefix_solve \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len 1827 --differ 205 --decode 16 --mb 440

# split
... --len 1827 --differ 205 --decode 0 --mb 440 --phases

# llama.cpp: rm the prompt cache, then cold/warm on llama-p1.txt and delta on llama-p2.txt
~/Workspace/llamacpp-arelath/build/bin/llama-completion \
  -m ~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base.f16.gguf \
  -c 2048 -t 4 -ngl 0 --no-warmup --no-display-prompt \
  --prompt-cache ~/models/nq-scratch/llama-f16.cache -n 16 --temp 0 --seed 1 \
  -p "$(cat "$P")"
```

## Review

Reviewed by `agy` with `gemini-4-hydrogen-high` on the combined tree. It refused the cold turn as
a dominant-term claim: `ffn` is the largest leaf after the tiling (37.4–37.6% against `attn.core`
34.3–35.1%), the pre-change 46.4% is from the `--phases` table rather than an executable
`phase_split --target` run, and the tiled cold spread (1.223×) is wider than the paired ratio. This
receipt already refuses the cold claim and reports the tail as the resolved result; its 1.627–1.686×
tail leaf is the claim. The `phase_split --target attn.core` gate at 54.5% is in `1795509414`, where
it is labelled context: it traces the decode path, not the tile.
