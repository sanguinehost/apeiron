# nq: the activation pack is a column-band dispatch

Criterion: **T2** (turn latency), partial. Grades: `measured` for the box walls, rates and phase
splits; `tested` for the pooled/per-column equality at every block; `model` for the serial-fraction
fit and the traffic the change does not touch. No 7B, no 20 tok/s, no decode claim, no laptop
claim.

Model: `skeleton-q8head.gguf` + `nq06-v3.nqp` (0.977441 b/w, 196 linears, 418,381,824 sign
terms/token, 28 layers, 8 KV heads × 128, 229,376 B/position charged), `--kernel q8act` (the
harness default, `examples/prefix_solve.rs:54`), `--len 1827 --differ 205 --decode 0`. Machine: OVH
box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan, `lscpu` reports 8 sockets × 1 core × 1
thread, 31.3 GiB, 256 MiB L3, no cpufreq governor), 8 participants, no GPU. Times UTC. The laptop,
the only device a timed claim may name, is **not measured**: `BAT0` 0%, 0 Wh, CPU pinned at
400 MHz. Every run under `~/bin/box-run` (agent-guard, 16 GiB cap, 8 jobs).

## Result

The pack moved off the caller. Box cold prefill, block 4:

| participants | before s | after s | wall ratio | projection sum before → after | ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 55.720 | 55.885 | 0.997 | 36.349 → 36.365 | 1.000 |
| 2 | 34.622 | 33.365 | 1.038 | 23.466 → 21.790 | 1.077 |
| 4 | 22.911 | 21.713 | 1.055 | 16.253 → 14.310 | 1.136 |
| 8 | 17.858 | 15.954 | **1.119** | 13.240 → 10.644 | **1.244** |

Rates at 8 participants: 100.55 → 114.52 tok/s (8.70 ms/token), block 8 123.46 tok/s (8.10
ms/token). The width-1 arm is the control: the same binary change costs 0.3% when no pool exists,
and the effect grows monotonically with the participant count, which is the signature of caller
work moved onto the pool. The dispatch count in the trace rises by exactly four per linear call
(`pool.wait` 4,773 → 11,045 = +6,272 = 4 × 196 × 8), so the new path ran.

This does not close T2. It removes one serial pass from the dominant leaf; the reader's term rate
and the attention core are unchanged.

## What changed

`project_pooled_block` built three things on the calling thread between its two pooled dispatches:

1. `t_j = round(s_pre_j·x_j / 2^32)` for every column and the `inn·max|t|` bound (a `k·inn` checked
   product walk),
2. the `t` and `z` activation quantizations, one `quant_act` per column (`k·inn` and `k·rank`
   elements, two passes each),
3. the `rank·max|z|` bound scan.

Each is now a column-band dispatch on the same pool, with the existing `cols = ⌈k/(4·width)⌉`
band:

- `TBuild` (`for_chunks2`) fills the flat `t` buffer's band and the matching `narrow_t` band; the
  per-column order, the checked `mul_q32` and the bound are the serial loop's.
- `quantized_columns` (`chunks_mut`) calls the same per-column `quant_act`; each column's `Option`
  is unchanged, so a refusal in a band is the same refusal.
- a `chunks_mut` scan fills `max_abs_z`; a max is order-independent.

Per-column order and every rounding are unchanged, so the result is bit-identical;
`the_pooled_block_is_bit_equal_at_every_block` steps `k` 1/7/33 against the per-column readers at
blocks 0–8 (`Reader::Exact`, `Q8`, `Ternary`) and passes. The exact and ternary readers still build
their masks serially (`quantized_columns` returns the empty slots with no dispatch), and the
residual's `x` reader is still on the caller; this pack has no residual layer, so that path costs
nothing here.

## Where the caller's time went, and what is left

Instrumented `project_pooled_block` (a scratch build with `Instant` timers around each step,
removed before commit), one 196-linear block at 8 participants, block 4, ms:

| step | ms |
|---|---:|
| `t` build (dispatch) | 27 |
| `t` quantize + stages | 98 |
| `V` stage (dispatch) | 200 |
| `rank·max|z|` (dispatch) | 3 |
| `z` quantize + stages | 52 |
| residual `x` stages | 7 |
| `U` stage (dispatch) | 395 |
| output gather | 126 |
| sum | 908 |

One sweep is 196 linears over one 228-column block; the cold turn is eight of them, so the table's
908 ms is per sweep and eight sweeps are 7.26 s of the 10.64 s projection leaf. The other 3.4 s is
the work inside those phases that is not `project_pooled_block` — SwiGLU, the q/k norm, the KV
append and the pass loop.

The output gather alone is 126 ms of the 908, and it allocates one `Vec` per column call — 44,688
allocations per sweep. It is the largest item the caller runs while the pool is idle: the fitted
serial share of `ffn` (0.18) is 1.16 s of 6.44 s at eight participants, 145 ms per sweep, and the
gather is 126 of it. The next levers, in order: the gather and `build_stages` (serial), the `U`
dispatch (395 ms/sweep; its per-row `project_row` epilogue makes it 1.45× the `V` dispatch per
term), and the reader's terms per cycle, which is where the 2.27× single-core gap to llama lies.

## The phase split at 8 participants

| phase | before ms | share | after ms | share |
|---|---:|---:|---:|---:|
| `ffn` | 7,904 | 44.3% | 6,441 | 40.4% |
| `attn.proj_in` | 3,680 | 20.6% | 3,122 | 19.6% |
| `attn.proj_out` | 1,656 | 9.3% | 1,082 | 6.8% |
| `attn.core` | 2,972 | 16.6% | 3,246 | 20.3% |
| `norm` | 269 | 1.5% | 620 | 3.9% |
| `residual` | 913 | 5.1% | 984 | 6.2% |
| `pool.wait` | 682 | 3.8% | 817 | 5.1% |
| cold wall | 17,858 | | 15,954 | |

`attn.core` and `norm` move the wrong way inside their run-to-run spread; the projection sum, which
is what the change targets, does not. Fitting `T(w) = s + (1−s)/w` to the arms: `ffn` 0.25 → 0.18,
`attn.proj_in` 0.31 → 0.24, `attn.proj_out` 0.33 → 0.17, `attn.core` ≈ 0.05 unchanged. The
remaining 0.17–0.24 is the gather, `build_stages`, the dispatch tail and the pre-dispatch setup.

## Head-to-head against llama.cpp

Same box, same window, alternating order, two reps each. llama.cpp `049326a` (build 9949) at
`-p 1827 -n 0 -r 2` on `Qwen3-0.6B-Base.f16.gguf` (1,192,230,912 B, 596,049,920 params; ubatch
512, its default); the engine at `--len 1827` on the same token stream (prefill block 256, its
default). llama's build carries **no AVX-512**: `GGML_AVX512=OFF` and `objdump -d
libggml-cpu.so` counts 0 `zmm` against 16,025 `ymm`, so both sides execute 256-bit AVX2 and the
per-core ratio is a like-for-like screen for the laptop, which has AVX2 only.

| participants | engine tok/s | llama F16 tok/s | wall ratio |
|---:|---|---|---:|
| 1 | 33.05 / 33.06 | 74.885 / 75.021 | 2.27 |
| 2 | 55.11 / 54.52 | 139.381 / 137.582 | 2.53 |
| 4 | 84.97 / 85.13 | 262.901 / 265.311 | 3.09 |
| 8 | 116.55 / 119.94 | 459.708 / 458.018 | 3.94 |

Scaling from one participant: engine 1.00 / 1.66 / 2.58 / 3.58, llama 1.00 / 1.86 / 3.52 / 6.13.
Fitting `T(t) = s + (1−s)/t` gives llama `s = 0.04` and the engine `s = 0.18`. The single-thread gap
is kernel efficiency; the widening gap is the caller's share of the sweep — the stages, the bound
scan and the gather are 286 ms of the 908 and do not scale.

The pack is arithmetic-neutral at prefill, so it cannot close that gap by itself. The ranks
(640/480/736) sit near the matrices' own dimensions: a 3072×1024 gate is `736·1024 + 3072·736 =
3.015M` MACs against `3072·1024 = 3.146M` dense. The pack's 54 MB against llama's 1.19 GB is a
byte property, and bytes bind decode, not prefill. With 418,381,824 terms/token the engine does
1.53 TFLOPs per 1,827-token prefill against llama's 2.18, and still takes 3.94× longer at eight
participants: the reader folds 32 terms per nine arithmetic instructions (fourteen with the loads
and pointers), and llama's F16 dot folds 8 MACs per convert-plus-FMA. Per core the engine's
projection leaf runs at 21.0 G terms/s, 42 GFLOPS, against llama's whole-prefill 89.5 GFLOPS at one
participant.

## Gates

The contributor tier on the box (`~/bin/box-run apeiron-nq`, agent-guard, 16 GiB, 8 jobs), tree =
`2f020b1c` plus this change; host `indra-dev-1`, rustc 1.99.0-nightly, cargo 1.99.0-nightly.

| command | result |
|---|---|
| `cargo test --workspace --locked --no-fail-fast` | **1099 passed, 3 failed, 83 ignored, exit 101** across 91 targets; the `--lib` line is 669 passed, 1 failed, 51 ignored. The 20 `engine::kern::nq` tests are 17 passed and 3 ignored (the timing harnesses), including `the_pooled_block_is_bit_equal_at_every_block`. The three reds are all `glslc`: `gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl` (`glslc` exit 1, `coopmat_gemm.comp:250` type error), and `tests/spirv_storage_race`'s `attn_coopmat_spirv_is_the_glsl` (`glslc` SIGSEGV) and `splitk_reduce_spirv_is_the_glsl` (the same compiler cannot reproduce the committed `.spv`, so the freshness check refuses) |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | exit 0 |
| `cargo fmt --all -- --check` | exit 0 |
| `python3 tools/check-claims.py` | exit 0 |
| `python3 tools/book-check.py` | exit 0, 20 glossary entries, 0 findings |
| `mdbook build book` + `mdbook test book -L $HOME/apeiron-mdbook/debug/deps` | both exit 0, from a clean `CARGO_TARGET_DIR=$HOME/apeiron-mdbook`; the shared `target/` is a mixed-feature directory and gives `E0464` |
| `cargo test --workspace --all-features --locked --no-fail-fast` | **1126 passed, 82 failed, 108 ignored, exit 101**, 25 targets. All 82 are the GPU tier: 60 panic at `src/gpu/vulkan.rs` (`vkCmdWriteTimestamp2 not found via vkGetDeviceProcAddr`) and 22 are `glslc` byte or `.spv` freshness checks. None is `nq`, and none is the pack path. |

The `--all-features` row is not comparable to the 681/45/62 in
`bench/cpu/1795518973-nq-blocking.md`: the default-tier composition is unchanged (669/1/51 on the
lib line), but the features now add 528 tests where they added 67 then, so the extra 445 passes and
37 failures are all in `gpu`, for the same two missing pieces — no `glslc` pin and no Vulkan device.
The tier `AGENTS.md` says runs on neither host stays red, and the `nq` suite is green in both tiers.

Every default-tier red is the same missing compiler. The local `glslc` is shaderc 2023.8;
`coopmat_gemm.comp` uses syntax it cannot compile, and `splitk_reduce.spv` cannot be reproduced from
its source, so the byte comparison and the freshness check both refuse. Neither touches the crate's
CPU path.

## What this does not cover

- **The laptop.** No timed claim until AC returns; the four core laptop (i5-10210U, 4 participants)
  is the device the ratio must be quoted on. A box ratio is a CPU-tier screen.
- **The maintainer gate.** No Aeneas, Charon or Lean on either host, so `proof/regen.sh --check`
  and `lake build` do not run and the committed extraction is not regenerated for this change. The
  blocked reader's twin remains `tested`, not `proved`; the change is in the safe pooled body, not
  in an `unsafe` block, and it adds no twin.
- **Speed.** 1.12× on the box at 8 participants is not a 20 tok/s or 30 tok/s claim and does not
  close the gap to llama.cpp. The reader kernel is unchanged.
- **The pack's carriers.** Only `Reader::Q8` dispatches its quantizer; `Exact` and `Ternary` still
  pack on the caller, and a pack with a residual layer still packs `x` there.
