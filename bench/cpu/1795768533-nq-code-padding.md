# NQ sign-dot: zero-padded activation codes remove the scalar tail

Criterion: **T1** prefill rate (instruction budget) and **T1** decode rate (no regression).
Grade: `measured` on the OVH box for the walls, instruction and cycle counts; `tested` for the
kernel equality tests. The laptop pair is open (`MemAvailable` below the 2 GiB guard). The timed
claim names the box; it is CPU-tier evidence, not laptop evidence.

## Result

The sign-dot kernels folded a scalar `code_select_twin` over `len % 32` terms on every row and
column. Zero-padding `CodeTerms::codes` to `32·⌈len/32⌉` lets all four AVX2 bodies run
`⌈len/32⌉` blocks and fold the final block vectorially. Prefill loses **3.1% of its
instructions**; cycles and wall do not move, so the scalar tail was overlapping with the vector
body, not binding it.

Paired binaries, same host, same model, alternating control and treatment:

| prefill 1,827 tok, reps 1 | wall ms | instructions e9 | cycles e9 |
|---|---|---|---|
| control `1193542b` | 3562.331 / 3599.900 / 3592.624 | 311.057 / 311.142 / 311.123 | 103.063 / 103.809 / 103.665 |
| padded | 3604.820 / 3626.004 / 3550.443 | 301.577 / 301.568 / 301.443 | 104.161 / 104.141 / 102.959 |
| means | 3584.95 / 3593.76 | 311.107 / 301.529 | 103.512 / 103.754 |

- Instructions: **−3.1%** (9.58e9 per pass).
- Cycles: +0.23%, wall: +0.25% — both inside the control's own spread (3.562–3.600 s, one
  treatment run at 3.550 s is the fastest of the six). The instruction step is claimed; the wall
  is not.
- Decode, three alternating `lookup_accept` runs each: control 5.908 / 5.798 / 5.924 ms/token
  (mean 5.877), padded 5.803 / 5.800 / 5.861 (mean 5.821). The padded arm is 0.9% faster on the
  mean and inside the control's 0.126 ms spread; the 28-round, 102/102 stream is exact in every
  run.

## Machine, binary and command

Machine: OVH box `ubuntu@139.99.223.161` (8 vCPU EPYC Milan, 32 GiB), `APEIRON_THREADS=8`,
`~/bin/adg`, `perf stat`. CPU-tier evidence; the timed target is the laptop (i5-10210U), which did
not run this measurement.

Model `skeleton-q8head.gguf`; pack `nq06-v3.nqp` (196 linears, 0.977441 b/w); tokenizer
`tokenizer.json`.

| arm | binary | sha256 |
|---|---|---|
| prefill control | `/tmp/prefill_control` | `c5d6ef3000c8906fee2a0ed77d0c61a2835ec7d5ebe496a20dabe7cbf182d1ad` |
| prefill padded | `/tmp/prefill_treat` | `d9c04b94cb379c62afe3e243bf1364e5f01e107f4a7d4803e84f7018e7e80c41` |
| decode control | `/tmp/lookup_control` | `d457cf361acd2409703816f7569c207653bd90b7424bed0605cc5e5548af6596` |
| decode padded | `/tmp/lookup_treat` | `4f3d7e1ec864469ff796c7e6417095abcc6d423bb7a888ed5ca3396f51b9b4cb` |

```sh
perf stat -e instructions,cycles -x, /tmp/prefill_{control,treat} \
  --model skeleton-q8head.gguf --pack nq06-v3.nqp --tokens 1827 --reps 1
APEIRON_THREADS=8 /tmp/lookup_{control,treat} \
  --model skeleton-q8head.gguf --pack nq06-v3.nqp --first 42 --n 128 --ngram 2 --draft 8
```

## Mechanism

`CodeTerms` now owns `codes: Vec<i8>` zero-padded to `blocks * 32`, and keeps the unpadded length
`n`. The four AVX2 bodies (`code_select_avx2`, `code_select_block_avx2`,
`code_select_block2_avx2`, `code_select_row_c_avx2`) compute `blocks = len.div_ceil(32)`, require
the padded plane (`codes.len() >= blocks*32`, `stride >= blocks`) and drop the tail fold. The
padded lanes multiply zero, so the selection is bit-identical; the safe twins and the scalar
dispatch keep the unpadded slice. The wide tile's padded-plane check stays in `StageCols::cols`
(group assembly), not in `selected_row_cols_ready` (once per output row): re-introducing the
per-row eight-column check cost 8.3e9 instructions when it was last tried
(`bench/cpu/1795660694-nq-select-sign-dot.md`).

## Phase split and ceiling

The target is the wide NQ row tile `code_select_row_c_avx2::<8>`, **27.67%** of prefill cycles on
the unmodified tree (`bench/cpu/1795765865-nq-prefill-single-run.md`, `perf record -e cycles:u`).
Its pipeline ceiling is `1/(1 − 0.2767) = 1.383×`. This change removes 8.7% of the tile's
instructions (`9.58e9` of `110.7e9`) and **0% of its cycles**; the measured wall share moved by
0. It is a kernel-rate step, not a wall step.

The prefill instruction budget is unchanged as a target: `1.982 s` at the measured
`77.9e9 instructions/s` needs `≤154e9` instructions, and this tree is at `301.5e9` for 1,827
tokens (0.90× of llama.cpp's 3.965 s, 1.80× short of half). A halving still needs a lower-MAC pack
or a wider-ISA host; this receipt claims neither.

## Scope

- Box only; no laptop timed row. The controller run is the OVH box.
- The equality tests (`the_code_source_equals_the_scalar_source`,
  `the_wide_source_equals_eight_per_column_sources`, `the_two_column_source_equals_two_per_row_sources`)
  pass; the full lib suite is 707 passed, 1 failed, and the failure is the known
  `coopmat_spirv_is_the_glsl` `glslc` byte check, not this change.
- Rust/LLVM codegen, the kernel ABI and the `unsafe` container are outside the claim.
- No wall or rate improvement is claimed for prefill or decode.

## Files

- `src/engine/kern/nq.rs`: padded `CodeTerms`, four bodies, group-assembly check, tests.
