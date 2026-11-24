# nq: the reader folds two activation columns per weight read

Criterion: **T2** (turn latency), partial. Grades: `measured` for the box walls, rates and phase
splits; `tested` for the two-column-vs-per-row equality over the vector domain; `model` for the
instruction count and the laptop's share of the result. No 7B, no decode claim, no 20 tok/s, no
laptop claim.

Model: `skeleton-q8head.gguf` + `nq06-v3.nqp` (0.977441 b/w, 196 linears, 418,381,824 sign
terms/token, 28 layers, 8 KV heads × 128, 229,376 B/position charged), `--kernel q8act` (the
harness default, `examples/prefix_solve.rs:54`), `--len 1827 --differ 205 --decode 0 --mb 440`.
Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan, 8 sockets × 1 core × 1 thread,
31.3 GiB, 256 MiB L3, no cpufreq governor), 8 participants, no GPU. Times UTC. The laptop, the only
device a timed claim may name, is **not measured**: `BAT0` 0%, CPU pinned at 400 MHz. Every run under
`~/bin/box-run` (agent-guard, 16 GiB cap, 8 jobs); the example was rebuilt with `touch src/lib.rs`
before the timed runs.

## Result

The blocked Q8 reader decodes a weight word once and folds it against two activation columns. Box
cold prefill, 8 participants, rotating order, three alternating pairs of the built-in default
(`NQ_BLOCK = 8`) against the previous default (`APEIRON_NQ_BLOCK=4`) on the same binary:

| arm | cold prefill s | tok/s |
|---|---:|---:|
| default (block 8, two-column) | 13.564 / 13.972 / 14.229 | 134.69 / 130.76 / 128.40 |
| `APEIRON_NQ_BLOCK=4` (two-column) | 15.695 / 15.621 / 15.626 | 116.41 / 116.96 / 116.92 |

Pairs: 1.157, 1.118, 1.098; mean **1.124**. Both arms run the pair, so this table measures the
row-block default, not the pair alone; the pair's own rate is in the next section. The same-window
dial screen, three reps each, cold prefill at 8 participants: block 4 15.542 / 15.704 / 15.638; block
5 15.148 / 15.093 / 15.286; block 6 15.117 / 14.751 / 14.183; block 7 14.367 / 14.348 / 14.381;
block 8 13.922 / 14.044 / 14.016 (before the default switch). Eight is the only arm with no
remainder block: rank 736 = 8·92, q out 2048 = 8·256, kv out 1024 = 8·128, gate/up out 3072 = 8·384,
down out 1024 = 8·128. Seven pays a one-row remainder on `V` and a six-row remainder on `U`, and the
one-row arm is the per-row path.

The projection sum (`ffn` + `attn.proj_in` + `attn.proj_out`) is **8,978 ms** at 8 participants on
the shipped configuration. The pack receipt's 10,644 ms is the previous shipped configuration
(block 4, single-column) in a different window; the two are not a pair, and the block-8 dial and the
pair both contribute to the drop.

## The default moves on box evidence, and the laptop row is open

`bench/cpu/1795518973-nq-blocking.md` kept `NQ_BLOCK = 4` because its laptop block-8 runs were
**inside** the block-4 arm's spread, not because 8 lost: block 8 was 2.3% below the bracketing mean,
the two block-4 runs differed by 2.8%, and the sequence was not alternating. This change moves the
default on the paired box screen above, with the new kernel. The earlier laptop pairs were made on
the single-column kernel and do not measure this one. The laptop confirmation is **open**; until it
runs, the block-8 row is `measured` on the box and `model` for the laptop. The register budget also
changes: the pair holds two accumulators per row, so `R = 8` needs sixteen plus the five constants,
and the box microbench shows block 8 below block 7. The cold turn still wins at 8 because the
production shapes pay no remainder block.

## What changed

`code_select_block_avx2` spends five vector instructions per weight word before any arithmetic on
the activations (broadcast, `vpsrlvd`, two `vpshufb`, `vpand`). The word does not depend on the
column, so `code_select_block2_avx2::<R>` decodes it once and runs the two
`vpmaddubsw`/`vpmaddwd`/`vpsllvd`/`vpaddd` chains against two stages. Per `(row, 32-term block)` the
pair costs 13 vector instructions against 18 for two single-column calls; with the per-row word load
counted, 14 against 20. Each column keeps its own codes, shift vectors and `e_min`, so its result is
the single-column result.

- `CodeTerms::selected_rows2` checks both readers' domains (`serves`), dispatches the pair, and on a
  refusal falls back to the per-row twin for each. `Stage::dot_rows2` admits the pair only when both
  stages are Q8 readers with no ternary form and `rows ∈ 2..=NQ_BLOCK_MAX`; otherwise the caller
  keeps `dot_rows` per column.
- the `V` and `U` stage loops walk adjacent columns in pairs and keep the single-column entry for
  the odd last column and for `rows = 1`.
- `NQ_BLOCK` moves 4 → 8, with the divisor argument and the disclosure above. `NQ_BLOCK_MAX` stays 8.

Every per-column order and rounding is unchanged, so each column is bit-identical:

- `the_two_column_source_equals_two_per_row_sources` steps `len` = 32, 33, 63, 64, 65, 127, 128,
  129, 1000, 1023 at `e` = 0, 20, 40; the two activations carry different `e_min`s and shift
  patterns, the stride is padded, the tail word's padding bits are set, and `rows` steps 2..=8. It
  compares `selected_rows2` against `code_select_rows_twin` for each column, and checks the pair
  entry refuses `rows` = 1 and 9 and a short output without writing either output. This is the test
  that exercises the vector loop; the small-shape pooled test below does not (`len/32 = 0`).
- `the_pooled_block_is_bit_equal_at_every_block` steps k = 1, 7, 33 (three pairs plus the odd
  column at k = 7) against the per-column readers at blocks 0–8 for `Exact`, `Q8` and `Ternary`.

## The reader rate by block

Ignored test `the_pooled_block_rate_by_block` (`cargo test --release --lib --locked
engine::kern::nq::tests::the_pooled_block_rate_by_block -- --exact --ignored --nocapture`, box,
width from `APEIRON_THREADS`), one `ffn` gate 3072×1024 at rank 736 and k = 228, mean of three
calls, G terms/s including the pack, stages and gather:

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| width 1 | 15.2 | 16.6 | 23.8 | 24.6 | 24.0 | 24.9 | 24.9 | 25.0 | 24.9 |
| width 8 | 61.4 | 63.7 | 90.7 | 94.7 | 90.5 | 108.7 | 110.4 | 118.7 | 101.2 |

The same test on the same tree with the pair disabled in a throwaway copy (`if false && cc + 1 <
ncols` in the two stage loops, then restored; `cargo test --release --lib --locked
engine::kern::nq::tests::the_pooled_block_rate_by_block -- --exact --ignored --nocapture`,
`APEIRON_THREADS=1` and `8`, log `/tmp/opencode/box-nq-single.log`): width 1 block 4 22.02, block 7
22.67, block 8 22.49; width 8 block 4 93.53, block 7 112.48, block 8 100.68. Against the pair, that
is **+9%** at width 1 block 4, **+10%** at width 1 block 7, **+5.5%** at width 8 block 7, **−3%** at
width 8 block 4 and **+0.5%** at width 8 block 8. The pair wins where the row block is 7 and is
neutral at 8, where the sixteen-accumulator spill is the limit; the cold default's 1.124× is
therefore mostly the row-block choice, with the pair neutral at the shipped block and positive at
one participant.

## Phase split at 8 participants

`--phases`, cold turn, one run (14.468 s; the run's own `pass` is 96.7% of the wall):

| phase | ms | share |
|---|---:|---:|
| `ffn` | 5,396 | 37.3% |
| `attn.core` | 3,188 | 22.0% |
| `attn.proj_in` | 2,613 | 18.1% |
| `residual` | 1,148 | 7.9% |
| `attn.proj_out` | 969 | 6.7% |
| `norm` | 633 | 4.4% |
| `pool.wait` | 428 | 3.0% |
| `embed` + `head` | 31 | 0.2% |

`ffn` is still the dominant term; the projection sum is 62% of the wall. `pool.wait` is 428 ms with
11,045 dispatch calls, against 817 ms in the pack receipt's window; the paired reader serves two
columns per call, so a column pass makes half as many calls, and each column's work is shorter
because the expansion is shared. The two runs are not paired, so the drop is directional.

Against llama.cpp `049326a` (F16, `-p 1827`, same box, no AVX-512), the head-to-head window before
this change put the engine at 116.55 / 119.94 tok/s and llama at 459.708 / 458.018 at 8
participants. This change's runs did not re-run llama, so the ratio is **not paired**: the shipped
defaults here are 128.40–134.69 tok/s, which is 3.4–3.6× llama's earlier number. That is still a
gap. The single-thread gap is not this change's target and was not measured separately in this
window; the microbench's width-1 arm gains 7% on the reader, not 2×.

## The device ceiling

The pair does not change the weight or activation traffic: the same bytes are read, and the pack
keeps its stored layout. The prefill projection's binding bound in `1795518973` was the instruction
stream, not bytes: 40.2 G terms/s at 10 instructions per 32 terms per row and the measured 3.14 G
instructions/s per core (4 cores, equal-IPC model), against a bytes bound of 183 G/s at the measured
22.33 GB/s and 349 G/s at the 42.656 GB/s spec peak. The pair removes 5 of 18 instructions per two
columns, so the same instruction budget admits 18/13 = 1.38× the terms. The measured rate at width 8
rises 9% at block 7 (108.7 → 118.7 G/s), so the loop is not at its issue bound; no `perf` counter was
read for the pair, and which port binds is open. The microbench's rates include the pack, stages and
gather, so they are not pure kernel rates.

## Mutation

One line of `code_select_block2_avx2` in a throwaway copy on the box, `acc_b[r] = ... sv_b` →
`... sv_a` (column b's block shift reads column a's vector):

| command | result |
|---|---|
| `cargo test --release --lib --locked engine::kern::nq::tests::the_two_column_source_equals_two_per_row_sources -- --exact` | **FAILED**, 1 failed: `assertion left == right failed: len 33 e 0 rows 2: column b` (`/tmp/nq-mutation-red.log` on the box) |
| same, after restoring the file | **ok**, 1 passed (`/tmp/nq-mutation-green.log` on the box) |

## Gates

Box contributor tier: the seven commands of `AGENTS.md` § Gates driven by `/tmp/box-gate.sh` on the
tree that carries this change, `rustc 1.99.0-nightly (87e5904f5 2026-07-20)`, `cargo
1.99.0-nightly (3efb1f477 2026-07-17)`, host `indra-dev-1`, log `/tmp/opencode/box-gate4.log`;
every command under `~/bin/box-run` (agent-guard, 16 GiB cap, 8 jobs):

| # | command | exit | result |
|---|---|---:|---|
| 1 | `cargo test --workspace --locked --no-fail-fast` | 101 | 1100 passed, 3 failed, 84 ignored. The three are `glslc` byte checks this host cannot run: `gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl` (the check that also runs with `gpu` off), `attn_coopmat_spirv_is_the_glsl`, `splitk_reduce_spirv_is_the_glsl` |
| 2 | `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | 0 | |
| 3 | `cargo fmt --all -- --check` | 0 | |
| 4 | `python3 tools/check-claims.py` | 0 | 46 graded claims; every one names a resolvable artifact |
| 5 | `python3 tools/book-check.py` | 0 | 20 glossary entries, 0 findings |
| 6 | `mdbook build book`, `mdbook test book -L …` | 0, 0 | |
| 7 | `cargo test --workspace --all-features --locked --no-fail-fast` | 101 | 1127 passed, 82 failed, 109 ignored. 60 panic at `src/gpu/vulkan.rs:1641` (`vkCmdWriteTimestamp2 not found via vkGetDeviceProcAddr`), 22 are `glslc` byte/freshness checks; all 82 are the GPU tier, which needs the compiler pin and a device this host does not have |

The default-tier failure at `gpu::nn::coopmat_gemm` is the one `AGENTS.md` names for the box: the
byte check runs with `gpu` off. No Rust, documentation or claim check is red.

## What this does not cover

- **The laptop.** The default switch and the pair are box-only evidence until AC returns; the i5's
  four participants and its AVX2-only pipeline are the device the ratio must be quoted on. The
  block-8 register pressure is identical in the ISA, but the laptop row is unmeasured.
- **The maintainer gate.** No Aeneas, Charon or Lean on either host. The new kernels are opaque in
  `proof/regen.sh` beside the single-column ones, their twin is the per-column
  `code_select_rows_twin`, and the equality above is `tested`, not `proved`.
- **rustc and LLVM code generation, the kernel ABI, Vulkan drivers and the GPU, and the `unsafe`
  container.** The change states nothing about any of them.
- **The exact and ternary readers.** Only the Q8 production reader is paired. `Exact` and `Ternary`
  keep the per-column path, and a pack with a residual layer still reads `x` on the caller.
- **The 7B.** The shapes here are the 0.6B's. The divisor argument for block 8 is specific to this
  pack's rank and output widths; a 7B needs its own dial screen.
