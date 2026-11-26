# nq: the prefill gap is execution rate, not instruction count

Criterion: **T2** (turn latency), partial: the cold 1,827-position CPU prefill of the 0.6B pack at
block 1024, width 8, the scope of `1795660694-nq-select-sign-dot.md`. This receipt changes no engine
code. It measures the per-column instruction and cycle cost of the committed build (`qavx18`)
against llama.cpp at `049326a00` on the same box in the same window, and sizes the levers that
remain. It is not a quality claim (the stream is unchanged, PPL `53.991898`), not a 7B result, not a
200k turn and not a decode result.

Grades: `measured` for every wall, count, share and branch rate below; `model` for the per-column
slope applied to 1,827 (the formula and the raw runs are given) and for every "instruction cut ⇒
wall" projection. No `proved` claim: no Lean, no extraction.

Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan, 8 sockets × 1 core × 1 thread,
31.3 GiB, 256 MiB L3, 2.4 GHz base, no AVX-512, no VNNI), no GPU. The laptop — the only device a
timed claim may name — is **not measured**. Every job under `~/bin/adg run --` (16 GiB cap, 8 jobs,
swap zero). The box is shared; the slope is the graded quantity and the wall is quoted beside it.

## The result

| | llama.cpp | ours | ratio |
|---|---:|---:|---:|
| instructions, 1,827 cascade columns | 327.1 G | **312.3 G** | **0.955** |
| cycles, same columns | 116.6 G | 129.4 G | 1.11 |
| IPC over that work | 2.81 | 2.41 | 1.17 |
| branch misses, 1,827 columns | 35.1 M | 347.9 M | 9.9 |
| cold-turn wall, printed | **4.004 s** | **4.861 s** | **1.21** |

**The instruction count is at parity and slightly better than llama's; the 1.21× wall is execution
rate.** 17% fewer instructions per column would also reach parity, but nothing in the profile needs
that: the gap is that ours retires 2.41 instructions per cycle where llama retires 2.81, and misses
10× as many branches. The residual between the cycle ratio (1.11) and the wall ratio (1.21) is the
effective clock, which this box does not hold constant across windows (3.3–3.8 GHz derived from
cycles ÷ wall ÷ 8; `model`).

## Commands and the slope

```sh
# llama.cpp 049326a00: -r 1 is one warmup plus one measured rep, so the difference is two prefills
perf stat -e cycles,instructions,branches,branch-misses \
  ~/llama.cpp/build/bin/llama-bench -m ~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base.f16.gguf \
  -p {64,1827} -n 0 -t 8 -r 1
# ours: --differ 0 makes the warm and shared turns full cache hits, so the run is one cold prefill
APEIRON_THREADS=8 APEIRON_PREFILL_COLS=1024 perf stat -e <the same counters> \
  ~/bin/prefix_solve.qavx18 --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len {64,1827} --differ 0 --decode 0 --mb 440
# shares: perf record -F 4000 -e instructions:u -o /tmp/line.data -- <the --len 1827 --differ 205
# line>, then perf report -i /tmp/line.data --no-children --sort symbol|srcline
```

| run | instructions | cycles | branches | misses | printed cold |
|---|---:|---:|---:|---:|---:|
| llama `-p 64 -r 1` | 17.937 G | 7.831 G | | 23.1 M | |
| llama `-p 1827 -r 1` | 649.261 G | 232.803 G | 42.695 G | 90.9 M | 4.004 s |
| ours `--len 64 --differ 0` | 32.798 G | 16.120 G | 3.473 G | 44.8 M | 0.223 s |
| ours `--len 1827 --differ 0` | 329.731 G | 139.120 G | 36.681 G | 375.5 M | 4.861 s |

The slope, `model`: llama's two runs differ by 1,763 columns twice, ours by 1,737 (1,827 − 64 cold,
plus the warm and shared turns' 3 against 16 — the cache holds 1,824 of 1,827 and 48 of 64).

| | instructions/column | × 1,827 | cycles/column | × 1,827 |
|---|---:|---:|---:|---:|
| llama | 0.17905 G | 327.1 G | 0.06380 G | 116.6 G |
| ours | 0.17095 G | 312.3 G | 0.07082 G | 129.4 G |

The 1.089 T MACs of the 0.6B at 1,827 positions divide out to 0.300 instructions/MAC for llama and
**0.287 for ours**. Our projection kernel is denser; the totals land at parity because ours also runs
the quantizer, the pooled dispatches and the separate epilogues that llama fuses.

`APEIRON_THREADS=1` is **not** a usable code measurement here: the same output costs 635.8 G
instructions against 372.0 G at width 8, so the one-thread path runs a different column shape. The
cause is not separated; the width-8 shape is the one the target runs.

## Where the instructions are

Symbol shares from `perf record` on the `--differ 205` line (whole process 371.98 G; `perf report
--sort symbol`). The G column is the share applied to 312.3 G (`model`, ±5%: the profile's
denominator holds the load and the shared turn).

| symbol | share | G |
|---|---:|---:|
| `code_select_row8_avx2` | 34.83% | 108.8 |
| `ColPipeline::run` | 15.46% | 48.3 |
| `gemm_tile<true>` (attention) | 9.73% | 30.4 |
| `gemm_tile<false>` | 8.11% | 25.3 |
| `attn_core_tile_avx2_inner` | 5.25% | 16.4 |
| `quant_act_grouped_avx2` | 3.29% | 10.3 |
| `swiglu_f32_avx2_8_into` | 2.78% | 8.7 |
| `rms_norm_fixed` | 2.60% | 8.1 |
| `rms_with_gain_slice` | 2.23% | 7.0 |
| `drop_glue::<Option<StepError>>` | 1.89% | 5.9 |
| `attn_head_group_tile_f32` | 1.64% | 5.1 |
| `pool::worker` | 1.64% | 5.1 |
| `code_select_rows2_avx2` | 1.48% | 4.6 |
| `rotate_pair_prefixes` | 1.47% | 4.6 |
| `qk_norm_heads` | 1.10% | 3.4 |
| `fixed_sin_cos` | 0.54% | 1.7 |
| `copy_finite` | 0.44% | 1.4 |

Rolled up against llama's symbol table (same window):

Both columns are the share applied to the cascade count of the table above (`model`); llama's
`projection` is `gemm_bloc<4,3>` 58.82% plus `fp16_to_fp32_row` 9.35%.

| | llama.cpp | ours |
|---|---:|---:|
| projection: weights read, unpack, dot | 223.0 G | 161.7 G |
| attention: scores, softmax, value fold | 94.2 G | 88.3 G |
| elementwise glue: quant, norm, silu, drop, pool | 9.9 G | 45.1 G |
| remainder (load, 32-token warm) | 0 | 17.2 G |

The composition is a trade, not a deficit: **our projection is 61 G cheaper than llama's and our glue
is 35 G dearer**, and the sum is 15 G in our favour. The projection's density is the committed
kernels':

| kernel | inner loop | MACs | instructions/MAC |
|---|---|---:|---:|
| llama `gemm<4,3,4>` | 19: 4 `vmovups`, 12 `vfmadd231ps`, `addq`, `cmpq`, `jne` | 96 | 0.198 |
| ours `code_select_row8_avx2` | 21: 4 expand, 8 `vpmaddubsw`, 8 `vpaddw`, `inc` | 256 | 0.082 |

Both from committed listings: `1795601655-llama-f16-4x3x4.s` (loop `.LBB1_38`) and the `perf
annotate` of `code_select_row8_avx2` at `nq.rs:1651…1711`. llama pays 30.2 G converting its f16
weights to f32; we pay 10.3 G quantizing the activation (a stage llama does not have) plus 45 G of
separate elementwise passes where llama's are fused.

## Execution rate

The branch gap is the one mechanism measured: 347.9 M misses against llama's 35.1 M, 1.02% of
branches against 0.21%. At ≈18 cycles each (`model`) ours loses 6.3 G cycles and llama 0.6 G, about
a quarter of the 12.8 G cycle gap; the rest is not separated here. The sign kernel is also
vector-port-bound by construction — 16 vector ALU ops in 21 instructions, so two Zen 3 vector pipes
cap it near 2.6 instructions/cycle — while llama's 12 FMAs in 19 instructions sit nearer 3.2.

## The reverted hoist, and what it says about resolution

`qavx19` (`0ebf58eb…`) hoisted the V epilogue's row scale and stride and sliced `t_build`'s three
views once per column. It is bit-identical (PPL `53.991898`) and cut instructions 371.7 → 363.5 G
(**−8.25 G, −2.2%**), but the wall read **0.999** over five interleaved pairs and **1.020** over
seven in a second window: the rate fell by the instruction cut. Reverted; HEAD is `2f830b2c` and
`qavx18` is its build. The consequence is a measurement rule for this host: **a change below ≈5% of
312 G (≈16 G) is inside the 2% layout noise and cannot be validated on the wall here.**

## The levers that remain

Line shares of the same profile, within the symbols above (`perf report --sort symbol,srcline`; G is
the line share applied to 312.3 G, `model`):

| lever | hot lines | G |
|---|---|---:|
| `U`/`V` epilogue checked `i128` (`project_scaled`, `mul_q32`) | `int_macros.rs:2553` (checked `add`) 2.71%, `nq.rs:2970` (`project_scaled` body) 2.28%, `cmp.rs:1050` 1.15%, `int_macros.rs:2765` (checked `mul`) 0.69%, `value.rs:11` (`round_shift_32`) 0.35% | 22.4 |
| norms: `i128` sum of squares + two roundings | `norm.rs:105/63`, `value.rs:11`, `num.rs:316`, `fixed_rsqrt.rs:45`; symbols 4.83% | 15.1 |
| the quantizer's scalar lane loop (4 `i64` codes stored, each range-checked) | `range.rs:885` 1.01%, `nq.rs:1272` 0.67%, `nq.rs:1230` 0.34%, `nq.rs:1271` 0.29% | 7.2 |
| the status vector's drop | `mod.rs:828` 2.03% (`softmax.rs:500/563`'s `Vec<Option<StepError>>`) | 6.3 |
| `swiglu`'s per-lane Q32↔binary32 | `fp.rs:336`, `avx.rs:48/283`, `cmp.rs:2031`; symbol 2.78% | 8.7 |

The five total **59.7 G, −19% of the wall**, which at the measured IPC and clock is 3.93 s against
llama's 4.00 s: parity, not a win. The first is the largest arithmetic block outside the dot itself
and is checked-lane overhead — `int_macros.rs:2553` is one `jo` per word. Its mechanism is llama's
(the scale applied in the epilogue, in registers); ours is scalar because the overflow admission is
per word. A band-level admission (the shape `narrow_z` already has for the source) or an eight-lane
exact `mul_q32` are the two ways to remove it.

**0.5× is refused at this algorithm.** With the glue at zero the projection and attention are
252.6 G, 3.93 s at the measured rates — 0.98× of llama, not 0.5×. Half of llama's 4.004 s is 2.00 s,
which needs the projection's instructions halved as well: a different attention (the cold context
tier) and a cheaper projection carrier, not a faster epilogue.

## Omissions

- No code changed: HEAD `2f830b2c`, tree clean, `qavx18` rebuilds bit-identical from it
  (`a2d3dff2…`, `nq.rs` `0f05e69c…`). No gate beyond that.
- The box is shared; the four raw runs are one window and were not interleaved with each other. The
  graded wall remains the interleaved pair in `1795660694-nq-select-sign-dot.md` (0.924 against HEAD
  `9215b81f`); this receipt's 4.861 / 4.004 s are single runs.
- The cycle differential is `perf stat` on a multithreaded process, ±2–4%; the clock is derived, not
  measured, and is why the cycle ratio and the wall ratio are both quoted.
- Our cycle count includes the pool's spin-before-park (`pool::worker` carries 1.64% of the
  instructions), which burns cycles while a participant waits; on a shared box that inflates the
  cycle count and understates our IPC. llama.cpp's OpenMP barrier does not spin. The wall ratio does
  not have this bias and is the primary number; the IPC row is an upper bound on the gap.
- `int_macros.rs`, `cmp.rs`, `range.rs`, `num.rs` and `mod.rs` line numbers come from the toolchain's
  debug info; the Rust source is not installed on the box, so the names are resolved from the stdlib
  paths, not read.
- The 17.2 G remainder is not decomposed below 0.44%.
- Nothing here is `proved`, and no Lean, Aeneas or Charon gate ran.
