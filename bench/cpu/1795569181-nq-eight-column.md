# nq: eight activation columns per weight read

Criterion: **T2** (turn latency), partial: the projection phases of the cold 1,827-position
prefill, not a turn, not decode. Grades: `measured` for the isolated rate, the cold walls and the
phase split; `tested` for the wide-vs-pair equality and the pooled bit equality; `model` for the
instruction count and the laptop. Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC
Milan, 8 sockets × 1 core × 1 thread, 31.3 GiB, 256 MiB L3, no cpufreq governor, no AVX-512), no
GPU. Times UTC. The laptop, the only device a timed claim may name, is **not measured**: the
window is not quiet and the box is shared (Omissions). Every run under `~/bin/box-run`
(agent-guard, 16 GiB cap, 8 jobs); the example was rebuilt with `touch src/lib.rs` before timed
runs.

Model: `~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf`; pack
`~/models/nq-scratch/nq06-v3.nqp` (196 linears, 28 layers, 418,381,824 sign terms/token, 0.977441
b/w), the harness default, `examples/prefix_solve.rs`, `--len 1827 --differ 205 --decode 0 --mb
440`.

## What changed

The pooled Q8 reader already read one factor row's weight word once per 32-term block and folded
it against two activation columns (`selected_rows2`). The tile widens that pass to **eight columns
at one factor row**:

- `ColRef` is one column's reader view (`codes`, `scales`, `shift_vecs`, `e_min`, `vector`) and
  `CodeTerms::col_ref` builds it; `selected_row_cols` checks eight views once, then calls
  `code_select_row8_avx2`.
- `code_select_row8_avx2` expands each weight word once (`vpsrlvd`, two `vpshufb`, `vpand`), then
  runs eight independent `vpmaddubsw`/`vpmaddwd`/`vpsllvd`/`vpaddd` chains, one accumulator per
  column. The two-column pair spends the same expansion once per pair; at eight columns the
  expansion is amortised over four times as many terms.
- `Stage::dot_row_cols8` admits the tile only when all eight stages are Q8 readers and no stage is
  ternary; a refusal leaves the caller's pair loop in place.
- `project_pooled_shape` runs whole groups of eight columns through the tile in **both** stage
  closures (`V` over `self.v` and `U` over `self.u`), then hands the leftover columns to the
  existing block loop at `wide_from`.
- `NQ_TILE_COLS = 8`; `APEIRON_NQ_COLS` selects the tile (`≥8`) or the pair (any value below 8,
  and a malformed value) for a same-binary A/B. Absent, the tile is the default at four or more
  participants and the pair below that.

The tile is a loop nesting, not an arithmetic. Each column keeps its own codes, shift vectors,
`e_min`, block sum and accumulation order, so the wide pass writes exactly the pair pass's words.
`the_wide_source_equals_eight_per_column_sources` asserts each of the eight outputs equals
`code_select_twin` over `len` = 32, 33, 63, 64, 95, 1000, 1023 with per-column exponents and shift
periods; `the_pooled_block_is_bit_equal_at_every_block` steps the whole pooled entry at `tile` =
2, 5, 8, 9 and `block` = 0…8 against the per-column entry (exact, q8 and ternary readers);
`the_wide_accumulator_agrees_when_the_bound_leaves_i64` drives the `i128` totals past `i64` and
back through the post-scale.

The direction is llama.cpp's: its quantised prefill interleaves activation columns in the repacked
blocks and folds a row block against them (`ggml_gemm_q8_0_4x8_q8_0`, `block_q8_0x4`,
`ggml/src/ggml-cpu/repack.cpp:2334`, `049326a`), where the single-column call is the decode shape.
This tree's carrier is the 1-bit sign factor, so the shared work is the word expansion rather than
an int8 block product; the loop property — one weight read, N activation columns — is the port.

## Result

### Isolated projection rate

Same binary, one `project_pooled_shape` call of the down shape (3072×1024, rank 736), 228 columns
= 0.687 G terms, block 8, per participant count; the rate is the block-8 row, mean of the test's
three timed calls, and the test asserts every tile/block output equals the tile-2 block-0 output
(`/tmp/opencode/box-cols.log`):

| participants | pair G terms/s | tile G terms/s | tile/pair |
|---:|---:|---:|---:|
| 1 | 25.28 | 21.92 | 0.867 |
| 2 | 43.97 | 42.03 | 0.956 |
| 4 | 67.47 | **80.09** | **1.187** |
| 6 | 108.56 | 110.73 | 1.020 |
| 8 | 106.85 | **130.38** | **1.220** |

The tile loses at one and two participants because eight accumulator chains offer half the
instruction-level parallelism of the pair's sixteen, and wins at four and eight where the expansion
is the larger cost. The six-participant row is nearly a wash; the pair is fastest at six
(108.56 G terms/s) while the tile is fastest at eight (130.38), so the two rows are not a monotone
curve.

### Cold prefill A/B

Same binary, alternating pair/tile, built-in defaults otherwise, `--decode 0`, no tracing; the cold
row of `prefix_solve` for a 1,827-token prompt (`/tmp/opencode/box-final.log`):

| participants | rep | pair s | tile s | pair/tile |
|---:|---:|---:|---:|---:|
| 4 | 1 | 18.168 | 17.969 | 1.011 |
| 4 | 2 | 18.301 | 18.028 | 1.015 |
| 4 | 3 | 18.081 | 17.999 | 1.005 |
| 8 | 1 | 13.092 | 13.200 | 0.992 |
| 8 | 2 | 13.432 | 13.041 | 1.030 |
| 8 | 3 | 13.149 | 12.995 | 1.012 |

Width 4, medians 18.168 → 17.999 = **1.009×**, paired mean 1.010, tile 3/3. Width 8, medians
13.149 → 13.041 = **1.008×**, paired mean 1.011, tile 2/3. The effect is at or inside the unchanged
arm's spread (width 4 pair 18.081–18.301 = 1.2%, tile 17.969–18.028 = 0.3%; width 8 pair
13.092–13.432 = 2.6%, tile 12.995–13.200 = 1.6%), so this table is a **sign test**, not a wall
gain: five of six pairs put the tile lower, and the isolated kernel ratio is 1.19–1.22×.

The traced A/B (`--phases`, `/tmp/opencode/box-cols.log`, `/tmp/opencode/box-cols-w4.log`) gives
the phase split and repeats the sign at width 4:

| participants | rep | pair s | tile s | pair/tile |
|---:|---:|---:|---:|---:|
| 4 | 1 | 18.326 | 17.853 | 1.027 |
| 4 | 2 | 18.184 | 17.930 | 1.014 |
| 4 | 3 | 18.195 | 19.058 | 0.955 |
| 8 | 1 | 13.126 | 12.950 | 1.014 |
| 8 | 2 | 12.884 | 13.026 | 0.989 |
| 8 | 3 | 12.902 | 12.982 | 0.994 |

Rep 3's width-4 tile arm is contaminated: `pool.wait` is 1,539.876 ms against 476–601 ms in the
other five arms and every phase moves with it; `ffn` is 8,453.776 ms against 7,783.770 / 7,843.522
in the two clean tile arms. Excluding it, the traced width-4 pairs are 1.014–1.027 and the width-8
pairs 0.989–1.014, the same ~1% sign as the no-trace run.

Clean width-4 phase split (rep 1; rep 2 in parentheses), pair → tile:

| phase | pair ms | tile ms | pair/tile |
|---|---:|---:|---:|
| `ffn` | 7,965.477 | 7,783.770 | 1.023 (1.023) |
| `attn.proj_in` | 3,370.140 | 3,315.517 | 1.016 (1.026) |
| `attn.proj_out` | 1,319.768 | 1,306.713 | 1.010 (0.997) |
| `pool.wait` | 484.400 | 475.927 | 1.018 |
| `pass` | 17,893.178 | 17,436.069 | 1.026 |
| cold wall | 18.326 s | 17.853 s | 1.027 |

Width 8 traced rep (`/tmp/opencode/box-trace-cols.log`): `ffn` 5,358.993 → 5,238.119 ms,
`attn.proj_in` 2,430.609 → 2,411.090, `attn.proj_out` 967.376 → 946.482, `pool.wait` 535.144 →
365.389, wall 13.137 → 12.910 s. The projection sum moves 1.9–2.3%, which is the kernel gain on
the projection's share of the phase; the rest of the wall does not move.

`perf` on the tile arm names the path: `code_select_row8_avx2` 11.0% of cycles and
`dot_row_cols8` 4.8%, against `code_select_rows2_avx2` 1.7% on the same run. The profile's largest
symbol is the pool worker's `pause` spin at 45.6% (model load and the serial phases), so those
shares are of a run that includes idle spin, not of the prefill's active time alone
(`/tmp/opencode/box-trace-cols.log`).

### Quality

Same binary, 2,047 positions, `--tokens 2048 --ctx 2048 --head-batch 16 --kernel q8act`,
`APEIRON_THREADS=8`, `/tmp/opencode/box-cols.log`:

| arm | PPL | mean NLL | first argmax |
|---|---:|---:|---:|
| pair | 572.461723 | 6.349946 | 315 |
| tile | 572.461723 | 6.349946 | 315 |

Identical to every printed digit. That is expected from the equality tests, not independent
evidence of quality: the tile changes no code path's arithmetic.

### Ceiling

The change targets the projection phases, and `ffn` (43.5% of the clean width-4 cold wall) is the
dominant leaf. An infinitely fast `ffn` moves that wall by at most `1/(1 − 0.435) = 1.77×`. The
no-trace cold-wall ratio is 1.009× (width 4, median) and 1.008× (width 8): a 0.9–0.8% move, 1.2%
of the 0.77 excess the ceiling admits, and inside the unchanged arm's spread. The isolated kernel
ratio is 1.187–1.220×, so the arithmetic improved and the wall barely followed: the projection dot
is not the whole of `ffn`, and the activation quantization, the `t`/`z` builds and the `mul_q32`
epilogues in the same phases do not shrink with the tile. This receipt is a kernel-rate step, not
the dominant-term fix; the next target is inside `ffn`'s non-dot share, measured before the
change.

## Commands

```sh
# cold prefill A/B, widths 4 and 8, three alternating pairs, no tracing
E="target/release/examples/prefix_solve \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len 1827 --differ 205 --decode 0 --mb 440"
for w in 4 8; do for rep in 1 2 3; do for cols in 2 8; do
  APEIRON_THREADS=$w APEIRON_NQ_COLS=$cols $E | grep '^cold'
done; done; done

# isolated rate, one process per participant count
BIN=$(ls -t target/release/deps/apeiron-* | grep -v '\.d$' | head -1)
APEIRON_THREADS=$w "$BIN" \
  engine::kern::nq::tests::the_pooled_block_rate_by_block --ignored --exact --nocapture

# PPL
P="target/release/examples/ppl \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --data ~/models/wiki.test.raw --tokens 2048 --ctx 2048 \
  --head-batch 16 --kernel q8act --dump-argmax"
APEIRON_THREADS=8 APEIRON_NQ_COLS=2 $P
APEIRON_THREADS=8 APEIRON_NQ_COLS=8 $P
```

Logs `/tmp/opencode/box-final.log` (no-trace A/B), `/tmp/opencode/box-cols.log` and
`/tmp/opencode/box-cols-w4.log` (rate and traced A/B), `/tmp/opencode/box-trace-cols.log`
(phase split and `perf`); scripts `/tmp/opencode/box-cols.sh`, `/tmp/opencode/box-cols-w4.sh`,
`/tmp/opencode/box-trace-cols.sh`, `/tmp/opencode/box-final.sh`.

## Gates

Contributor tier on the box copy (`/tmp/opencode/box-final.log` and the post-fix rerun
`/tmp/opencode/box-gate2.log`; host `indra-dev-1`, rustc 1.99.0-nightly):

| step | command | result |
|---|---|---|
| 1 | `cargo test --workspace --locked --no-fail-fast` | box-gate2: 674 passed, 1 failed, 53 ignored in the lib; 1,104 passed, 3 failed, 85 ignored across all targets. The three are the box's shaderc 2023.8 byte checks: `attn_coopmat_spirv_is_the_glsl` in the lib and in a four-test target, and `splitk_reduce_spirv_is_the_glsl`. No Rust logic test failed. |
| 2 | `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | box-final failed on `clippy::match_same_arms` in `parse_cols` (the explicit below-tile value and the absent low-width value both returned 2); merged to one wildcard arm, box-gate2 exit 0 (9.69 s). |
| 3 | `cargo fmt --all -- --check` | box-gate2 exit 0 |
| 4 | `python3 tools/check-claims.py` | box-final exit 0; local rerun after the book and receipt edits exit 0 |
| 5 | `python3 tools/book-check.py` | box-final exit 0; local rerun 20 glossary entries, 0 findings |
| 6 | `mdbook build book` + `mdbook test book` | box-final build and test exit 0; local `mdbook build book` exit 0 after the book edits (no code block changed) |
| 7 | `cargo test --workspace --all-features --locked --no-fail-fast` | 1,131 passed, 82 failed, 110 ignored; every failure is the GPU/Vulkan set (`GL_EXT_bfloat16` unsupported by the box's shaderc, `glslc` byte comparisons, no device), the same set as the gate baseline |

The three shaderc reds and the 82 all-features reds are environmental: the box has no GPU and its
shaderc 2023.8 does not reproduce the committed SPIR-V. The post-fix tree differs from box-final
only by the `match_same_arms` merge, which is behavior-identical; box-gate2 reruns steps 1–3 on it
and the local checks cover the documentation edits.

## Omissions

- The laptop timing is not measured. The box is Zen 2 with 256-bit AVX2 and no AVX-512; the
  laptop is Comet Lake with the same AVX2 width and two 256-bit FMA ports, so the kernel ratio is
  a `model` for the laptop until a quiet-window run. The box window is shared: a second lane holds
  the heavy lock between runs, and the width-4 tile rep 3 shows a 3.2× `pool.wait` spike.
- The maintainer gate (`proof/regen.sh --check`, `lake build`) does not run on either host. The
  new `code_select_row8_avx2` is added to `SERVING_OPAQUE` with `code_select_twin` as its
  per-column twin, and `proof/README.md` carries it in the debt list; the extractor's acceptance
  of the entry is unverified.
- The GPU/SPIR-V tier is not run (no device, no `glslc` pin).
- Decode is untouched: the tile needs `k > 1`. The head is untouched: it does not take
  `project_pooled`.
- The 8B is not measured. Its `out`, `inn` and rank change the row counts and the group of eight
  columns; the body handles any multiple of 32 with a scalar tail.
- The width-4 PPL is not a separate row; the arithmetic is per column and the width-8 A/B is
  identical, so a width-4 PPL would repeat the same words. The width-4 cold wall is measured.
