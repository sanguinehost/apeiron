# nq: the select kernel's sign dot, and the pooled admission

Criterion: **T2** (turn latency), partial: the cold 1,827-position CPU prefill of the 0.6B pack at
block 1024, width 8. Every value below is bit-exact against the same stream (PPL `53.991898`), so
this is not a quality claim. It is not a 7B result, not a 200k turn and not a decode result.

Grades: `measured` for every wall, paired ratio, PPL, instruction count and phase split; `tested`
for the equality tests and the mutations. No `proved` claim: the changed bodies sit in the
extraction closure (`kern/nq.rs`), the committed extraction predates them and the Lean bridge did
not run here (Omissions). Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan,
8 sockets × 1 core × 1 thread, 31.3 GiB, 256 MiB L3, 2.4 GHz base, no AVX-512, no VNNI), no GPU.
The laptop — the only device a timed claim may name — is **not measured**. Times UTC; every job
under `~/bin/adg run --` (16 GiB cap, 8 jobs, swap zero).

## Commands

```sh
# cold prefill, width 8, block 1024
APEIRON_THREADS=8 APEIRON_PREFILL_COLS=1024 \
  ~/bin/prefix_solve.<arm> --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len 1827 --differ 205 --decode 0 --mb 440
# the same line under perf stat -e instructions, and the same line plus --phases for the split
# PPL, same stream and window for every arm
~/bin/ppl.<arm> --model <same gguf> --pack <same nqp> --tokenizer <same json> \
  --data ~/models/wiki.test.raw --tokens 8192 --ctx 512 --head-batch 16 --kernel q8act
# llama.cpp at 049326a00, same host and window
~/llama.cpp/build/bin/llama-bench -m ~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base.f16.gguf \
  -p 1827 -n 0 -t 8 -r 3
# symbol shares: perf record -F 4000 -e instructions:u -o <data> -- <the prefill line>, then
#   perf report -i <data> --no-children --sort symbol --percent-limit 0.5
```

Model: `skeleton-q8head.gguf`; pack `nq06-v3.nqp` (196 linears, 28 layers, 0.977441 b/w, `k = 0`
salient, `res_layers = 0`). Every timed build follows `touch src/lib.rs` and runs under
`~/bin/adg run --`. The arms are `qavx7` (`e26e47a6…`, HEAD `9215b81f`), `qavx8` (`1b2bae7f…`),
`qavx9` (`b96015be…`), `qavx10` (`d5e55aaf…`, reverted norm), `qavx13` (`b96015be…`, rebuild of
`qavx9`'s source), `qavx14` (`a95d392a…`, the admission arm), `qavx17` (`a80d7683…`, the final
arm of the first window) and `qavx18` (`a2d3dff2…`, the build of the committed source).

## What changed

HEAD is `9215b81f`. Three dials on the one `matmat_q8act` path, none a second projection function.

### `project_scaled`: the narrow epilogue (`qavx8`)

`src/engine/kern/nq.rs`. The pooled `U` stage called `project_row` per `(row, column)`, whose dense
arm forms `round(dot·s_post / 2^32)` in `i128` with a checked product, a rounding add and a checked
narrowing. `project_scaled` takes the same value in `i64` whenever the dot, the product and the
rounding term all fit — one 64-bit multiply, one add, one shift — and falls through to the `i128`
body for everything else, so the narrow arm shortens the path without narrowing the domain. It is
inlined into the tile, so the shared entry's sparse and residual loops leave the hot loop.

### The select kernel: the sign dot in the fold (`qavx9`)

`code_select_row8_avx2` returned each column's selection `Σ_b S_b·Σ_{bit=1} codes`, shifted by
`e_min` into `i64`; `StageCols::dot_row` then made a second pass over the eight results to form
`sum − 2·selected` in `i128` with checked arithmetic. The kernel now carries `sum` on `ColRef`,
forms the sign dot in its own epilogue (`sum.checked_sub(selected.checked_mul(2)?)?`), and returns
the eight sign dots directly:

- the per-column shift is `wrapping_shl(e_min)` in `i64`, exact because `CodeTerms::new` admits only
  `scale_sum·4064 ≤ i64::MAX`, so `e_min < 64`;
- `lane_sum` is the five-instruction fold (`vpaddd` of the halves, two `vphaddd`, `vmovd`), exact
  because the reader admits only `norm·4064 ≤ i32::MAX`, which bounds every lane, both halves and
  the eight-lane total;
- a subtraction that would leave `i64` returns `None`, and the caller's exact `i128` loop forms the
  same value, so the refusal route is unchanged.

The safe twin is `code_sign_dot_twin`: the selection from `code_select_twin` narrowed exactly as the
kernel narrows it, then the same checked `sum − 2·selected`, `None` on the same boundary. The
equality tests (`the_wide_source_equals_eight_per_column_sources`,
`the_wide_source_folds_equal_block_runs`, the pooled tile) compare against it over whole blocks, a
padded tail and a ragged group. Mutations: `checked_sub` → `checked_add` at the kernel's epilogue
reddens 5 tests; removing the tail's `checked_add` reddens the tail cases; restoring either turns
them green.

### The pooled admission (`qavx14`)

`StageCols::cols8` checks each column's `len ≤ codes.len()`, `shift_vecs.len()` and `scales.len()`
once, when the group of eight is assembled, instead of once per output row; the admitted `len` is
stored on the group and `dot_row` no longer takes one, so a row cannot be folded against a length
the columns were not checked against. `scale_bound` derives the narrow-source bound from the
quantizer's group scales (`127·max S ≥ max|x|`, at most twice the true maximum; an over-estimate
only sends the column to the exact reader) instead of a second pass over every word of `z` and `x`.

### The lane bound, corrected before commit (`qavx17`)

The first form of the five-instruction fold was admitted under `norm·508 ≤ i32::MAX`, which bounds
one lane, not the eight-lane total the fold adds; a column with `508·norm ≤ i32::MAX < 4064·norm`
could wrap. The admission is now `norm·4064 ≤ i32::MAX` — the total, which implies the per-lane
bound — and `the_reader_refuses_a_lane_total_that_leaves_i32` pins the refusal
(`scales = [1, 2^20]`, `norm = 2^20 + 1`, `4064·norm > i32::MAX`). Real columns have `norm ≤ 96` on
this pack, so the reader is unchanged on the measured path; a refused column keeps the exact path,
which is the same arithmetic. `lane_sum`'s doc, `code_select_block_avx2`'s bound and the kernel's
`# Safety` now state the total, not the lane.

### Reverted: the fused in-place norm

`qavx9` → `qavx10` replaced `rms_norm_fixed` + gain with one in-place body over the head. It
measured **4.634 → 4.720 s, mean pair 1.018**, all five pairs > 1.000, PPL identical. The old path's
per-element normalization is vectorized inside `rms_norm_fixed`; the fused form redid it scalar.
Reverted, and not in the diff.

`qavx10`–`qavx12` also exposed a box-provenance trap: the box is a copy without `.git`, and
`box-push` sends only the files `git status` reports, so the `git checkout` that reverted
`loader.rs`/`step.rs` never reached the box and `qavx11`/`qavx12` were built from a tree that still
carried the fused norm. `qavx13` is the rebuild after an explicit `scp` of the four files; it
hashes **`b96015be…`, identical to `qavx9`**, and that hash identity is the provenance check for
`qavx9`'s numbers. `qavx10`'s norm pair was measured against `qavx9` and is unaffected. Filed as
friction `f-264a181aca86`.

## Result

### Against HEAD: five interleaved rounds in one window

| round | `qavx7` (HEAD) s | `qavx17` s | pair |
|:---:|---:|---:|---:|
| 1 | 5.060 | 4.844 | 0.957 |
| 2 | 5.018 | 4.791 | 0.955 |
| 3 | 5.077 | 4.678 | 0.921 |
| 4 | 5.007 | 4.656 | 0.930 |
| 5 | 5.079 | 4.775 | 0.940 |
| mean | **5.048** | **4.749** | **0.941** |

Every pair moves the same way. The control's spread is 5.007–5.079 (1.4% of its mean), the change's
4.656–4.844 (4.0%); the mean of the five pairs is 0.9407 and the ratio of the means is 0.9407. The
rate is 362 → 385 tok/s at width 8.

### The committed build, confirmed in its own window

The working tree's last edit before the commit is a `dead_code` reason on the twin and a doc line;
both shift panic locations and the binary's layout, so `qavx17`'s hash no longer names the
committed source. `qavx18` is the committed source's build (`a2d3dff2…`, `nq.rs`
`0f05e69c…`, hot kernel byte-identical), measured against the same HEAD control in a second
window:

| round | `qavx7` (HEAD) s | `qavx18` s | pair |
|:---:|---:|---:|---:|
| 1 | 5.318 | 4.768 | 0.896 |
| 2 | 5.044 | 4.680 | 0.928 |
| 3 | 5.082 | 4.757 | 0.936 |
| 4 | 5.414 | 4.982 | 0.920 |
| 5 | 5.043 | 4.741 | 0.940 |
| mean | **5.180** | **4.786** | **0.924** |

The control's spread in this window is wider (5.043–5.414, 7.2%), so its mean is not comparable to
the first window's; the pair is. Instructions: `qavx18` 372.5 G against `qavx17` 371.9 G, a 0.2%
layout difference, and the committed source is what the next receipt builds on.

### The sub-arms, each paired in its own window

| arm | rounds | control s | change s | pair | binaries |
|---|---:|---:|---:|---:|---|
| `project_scaled` epilogue | 3 | 5.043 | 4.842 | 0.959 | `qavx7` → `qavx8` |
| select-kernel sign dot | 5 | 4.818 | 4.632 | 0.962 | `qavx8` → `qavx9` |
| pooled admission | 5 + 5 + 7 | 4.705 / 4.813 / 4.815 | 4.634 / 4.780 / 4.787 | **0.985 / 0.993 / 0.994** | `qavx13` → `qavx14` |
| fused in-place norm (reverted) | 5 | 4.634 | 4.720 | 1.018 | `qavx9` → `qavx10` |

The admission arm's per-round values, in window order:

```text
4.757 4.734 4.760 4.641 4.632  vs  4.732 4.642 4.538 4.580 4.676
4.797 4.746 4.930 4.717 4.877  vs  4.718 4.800 4.794 4.766 4.823
4.773 4.811 4.767 4.787 4.795 4.846 4.928  vs  4.779 4.710 4.713 4.740 4.840 4.828 4.902
```

Each window has its own absolute walls; they are not summed or compared across rows, and the
product of the sub-arm pairs is not the graded pair. The admission arm is the combined 17 pairs:
**0.991**, 12 of 17 below 1.000, with a 2.2% instruction cut. It is kept for the instruction cut
and the consistent direction, not as a wall claim: its 0.9% wall response is small against the
per-pair spread.

### Instructions

| arm | instructions per prefill | wall in that window |
|---|---:|---:|
| `qavx8` | 396.5 G | 4.825 s |
| `qavx9` | 379.9 G | 4.658 s |
| `qavx13` (rebuild) | 380.2 G | 4.757 s |
| `qavx14` | 371.9 G | 4.732 s |
| `qavx17` (final) | **371.9 G** | 4.710 s |
| llama.cpp F16, `-r 3` | 324 G | 3.829 s |

The counts are `perf stat -e instructions` on a quiet box; under contention the pool's spin before
park inflates them (435 G for `qavx7` in the final window, 451 G for the same binary at load 7.6),
so HEAD's loaded count is not comparable and the chain above is the measured one. The final count
is 47.9 G above llama's 324 G (1,295 G over the warmup plus `-r 3`); at each binary's own rate —
ours 78.9 G/s, llama's 84.6 G/s — that is the 0.92 s wall gap, so the gap is both instructions and
rate.

Symbol shares of the sampled instruction stream on `qavx9` (365.9 G of samples; the select kernel's
share fell from 37.69% on `qavx8` to 32.85%):

| symbol | share | G |
|---|---:|---:|
| `code_select_row8_avx2` | 32.85% | 120.2 |
| `ColPipeline::run` | 15.95% | 58.4 |
| `gemm_tile::<true>` | 9.47% | 34.6 |
| `gemm_tile::<false>` | 7.94% | 29.1 |
| `attn_core_tile_avx2_inner` | 5.14% | 18.8 |
| `quant_act_grouped_avx2` | 3.30% | 12.1 |
| `swiglu_f32_avx2_8_into` | 2.74% | 10.0 |
| `StageCols::dot_row` | 2.53% | 9.3 |
| `rms_norm_fixed` | 2.40% | 8.8 |
| `rms_with_gain_slice` | 2.22% | 8.1 |
| `drop_glue::<StepError>` | 1.80% | 6.6 |
| `KvState::attn_head_group_tile_f32` | 1.63% | 6.0 |
| `rotate_pair_prefixes` | 1.51% | 5.5 |
| `pool::worker` | 1.45% | 5.3 |
| `code_select_rows2_avx2` | 1.44% | 5.3 |
| `qk_norm_heads` | 1.07% | 3.9 |
| `fixed_sin_cos` | 0.57% | 2.1 |

The next terms are `ColPipeline::run`'s own checked epilogue and `t_build`, `quant_act_grouped_avx2`
and `swiglu`; the select kernel's per-column `vpmaddubsw` + `vpaddw` chain is at its floor on this
host (no VNNI, no AVX-512).

### Quality

| arm | PPL | mean nll | tokens |
|---|---:|---:|---:|
| `qavx7` (HEAD) | 53.991898 | 3.988834 | 8,176 |
| `qavx9` | 53.991898 | 3.988834 | 8,176 |
| `qavx10` (reverted norm) | 53.991898 | 3.988834 | 8,176 |
| `qavx17` (final) | **53.991898** | **3.988834** | 8,176 |

Same stream, 16 windows of ≤ 512, `--kernel q8act`. Every change is word-exact on the values it
touches; no PPL movement is claimed or observed.

### Phase split

`--phases`, one traced cold run of the final binary in a quiet window: 4.542 s traced (the graded
window's untraced mean is 4.749 s in a different window).

| phase | ms | share |
|---|---:|---:|
| `ffn` | 1584.909 | 34.9% |
| `attn.core` | 1434.133 | 31.6% |
| `attn.proj_in` | 594.254 | 13.1% |
| `attn.proj_out` | 246.240 | 5.4% |
| `norm` | 148.949 | 3.3% |
| `residual` | 121.722 | 2.7% |
| `embed` | 9.462 | 0.2% |
| `head` | 5.635 | 0.1% |
| `pass` | 4178.346 | 92.0% |
| `pool.wait` | 255.675 | 5.6% |

The pre-change split is HEAD's own (`fa6e18b2`; `1795653483-nq-prefill-avx2-attn-norm.md`, 5.152 s):
projection leaves `ffn` 36.7% + `attn.proj_in` 14.6% + `attn.proj_out` 5.7% = **57.0%** of the run
(`1/(1 − 0.570) = 2.33×`), `attn.core` 28.9% (1.41×). This change's target is the projection family,
whose select kernel was the largest single symbol at 37.7% of instructions on `qavx8` (its own
ceiling 1.61×). After the change the projection leaves are 2425.403 ms, **53.4%** of the traced run
(2.15×), still the dominant family over `attn.core` 31.6%. `examples/phase_split --target` measures
a decode token; this receipt's phase is a prefill leaf, so the dominant-term gate is applied with
`prefix_solve --phases` on the unmodified tree.

### Against llama.cpp

llama.cpp at `049326a00`, same host and window, `-p 1827 -n 0 -t 8 -r 3`:

| window | llama tok/s | llama s | our s | ratio |
|---|---|---:|---:|---:|
| final (`qavx17`, 5 rounds) | 477.18 (three reps: 478.02, 477.39, 476.14) | 3.829 | 4.749 | **1.24×** |

The target is 1×–0.5×: parity needs the wall to fall a further **1.24×** (0.92 s), half a further
**2.48×**. The remaining gap is the projection pass's own epilogue and staging (`ColPipeline::run`,
15.95%), the attention GEMMs (17.4% combined) and `attn_core` (5.1%), not the select kernel.

## Review

`agy --dangerously-skip-permissions -p … --model gemini-4-hydrogen-high` on the diff and this
receipt. It refused: (1) the admission arm as a wall change (0.991 inside the per-pair spread) —
the receipt says it is kept for the instruction cut, not the wall; (2) the first `lane_sum` fold,
which was admitted under a per-lane bound and could wrap the eight-lane total — corrected by the
`norm·4064` admission and the refusal test above; (3) `StageCols::dot_row` taking a `len` the group
had not checked — corrected by storing the admitted `len` on the group; (4) the kernel having no
non-test twin for `sum − 2·selected` — `code_sign_dot_twin` is now in the crate and is the tests'
specification; (5) `phase_split --target` not run — recorded above as the prefill/decode
instrument difference, with the pre-change share and ceiling quoted. It also corrected `1.21×` to
`1.22×` on the earlier window (the final window reads `4.749/3.829 = 1.240`), and noted the
round-4 quotient. The review is critique, not evidence; its four code findings are resolved in the
diff and the numbers are this receipt's own.

## Omissions

- The laptop is not measured. The box is shared; absolute walls drift across windows and the
  interleaved pair is the graded number. The phase split is a different window from the graded pair
  and its absolute wall is not comparable to it.
- The extraction and bridge are stale: `proof/regen.sh --check` has not run since `57b4352d`, no
  pinned Aeneas/Charon/Lean is installed on either host, and the changed bodies (`project_scaled`,
  `selected_row_cols_ready`, `scale_bound`, the `ColRef` sign-dot epilogue) are in the Charon
  closure. `code_select_row8_avx2` and `lane_sum` are declared `--opaque`; the kernel's twin
  `code_sign_dot_twin` calls only safe code and is named in `regen.sh`'s twin comment. The changed
  bodies are `tested`, not `proved`; the maintainer gate inherits them.
- `cargo test --workspace --all-features` needs the `glslc` pin and a GPU device; neither host has
  them. On the box the working tree (nq.rs `267514f5db55d6598190b1dd01716a9e`) is green on
  `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`,
  `tools/check-claims.py`, `tools/book-check.py`, `mdbook build book` and `mdbook test book`;
  `cargo test --workspace --locked` is **689 passed, 1 failed, 54 ignored**, and the failure is
  the documented
  `gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl` byte check, which needs `glslc`.
- The sub-arm rows are four windows; only the final row is interleaved against HEAD in its graded
  window. The admission arm's 0.991 is a mean over 17 pairs and is reported as small.
- The `qavx8` and `qavx9` rows' instruction counts were taken in windows of their own; the deltas
  are quoted against those counts, not against the final window. HEAD's 435 G in the final window
  is spin-inflated and is not used as a baseline.
- No decode, turn, 7B, context or GPU claim.
