# One path: the fold is a dial transplant, not a deletion

Criterion: **T1** (decode) and **T2** (turn latency), through the architecture rule in `AGENTS.md`
("One path"). Grade: `model` for the plan. Every geometry, call site and test named below is read
from the source at the cited line; no timing is claimed. Scope: the three entries
`book/src/engine/onepath.md` names as debt — the pooled `Q8_0` VNNI `gemm_q8_0`, the pooled `Q4_K`
`gemm_q4_k` and the pooled `Q6_K` `gemm_q6_k`. This receipt changes no code.

## 1. The obvious fold is a measured regression

`matmat_q8act`'s `pooled` loop (`src/engine/kern/matmat.rs:292`) calls one row kernel per row:
`q8_0_row_kcols_avx2` or `q5_0_row_kcols_avx2` (`src/engine/chart/mod.rs:430,508`). Each decodes
four 32-lane groups once per row and folds them against every column with `dot32x4_codes`. That is
the `Q5_0` prefill fix, and it is correct. It is not a superset of the three debt entries, and the
difference is throughput, not arithmetic:

| dial | `gemm_q8_0` (`src/engine/kern/gemm.rs:498`) | shared row kernel | delete-first effect |
|---|---|---|---|
| activation panel | transposed once per call, 8 columns/group | per-column `i8` codes read per row | panel rebuilt per row |
| columns per lane | 8 | 4 + horizontal `reduce4` | half the lanes, extra reduce |
| row tile | 2 rows per pass, shared activation loads | 1 row | activation loads double |
| accumulator | power-of-two `i64`, one rounding | exact `i128` per block, one rounding | wider, slower epilogue |
| admission | AVX-VNNI, power-of-two scales, `n_in ≤ 512` blocks | AVX2 twin, any admitted width | narrower domain is fine; the twin must stay |

`gemm_q8_0` is the gateway for every `matmat_q8act` `Q8_0` call with `k > 1`
(`src/engine/kern/matmat.rs:156`), and `gemm_q8_0_group` (`gemm.rs:818`) and
`gemm_q8_0_swiglu_carrier` (`gemm.rs:955`) share its pack. Its output equals the per-column contract
on the narrower domain, pinned by `gemm_equals_the_per_column_contract` (`gemm.rs:2608`). The value
tests that survive a deletion — `columns_match_per_column_matvec_q8act`
(`src/engine/kern/matmat.rs:1007`), `tests/engine_tap.rs`, `tests/engine_f16_limb.rs` — would stay
green. The regression would be invisible.

On this laptop the entry never runs: the i5-10210U reports `avxvnni = false`, and `gemm_q8_0`
refuses without it (`gemm.rs:517`). `pooled_against_gemm_q8_0`
(`src/engine/kern/matmat.rs`, `--ignored`) prints `avxvnni false` and skips; the shared loop is
already the only `Q8_0` `k > 1` route here. The packed entry is an AVX-VNNI-host dial, so its
transplant can only be timed on such a host, not on this one.

`Q4_K` is the same shape with two more dials:

| dial | `gemm_q4_k` (`src/engine/chart/kcols.rs:1406`) | shared row kernel | delete-first effect |
|---|---|---|---|
| row decode | `Q4KRow` decodes a whole row once, reused by every column group | block scales re-decoded per 32-column group | decode repeated per group |
| lane selection | packed lane with per-column `emin`/`ke`/`spread`, `lane_spread_limit` | exact `i128` fold only | the common fast lane is lost |
| xsum | `Σ xq` per 32-lane group, `i32`, `k·n_in/8` B | none | new vector the shared loop must carry |
| task plan | time-sliced `run_global`/`ColWrites` into `k` columns | `for_chunks` over an interleaved `n_out×k` stage | one interleave copy, coarser tasks |
| budget | none | `k ≤ 4096`, `n_out·k ≤ 2^22` | out-of-budget calls fall to `per_column` |

The `Q4_K` row kernel the shared loop could call, `q4_k_row_kcols_avx2` (`kcols.rs:185`), is the
exact-`i128` fold. Routing `Q4_K` through it is bit-equal — and slower on the path the test treats
as common (`src/engine/chart/tests.rs:1165` requires zero `i128` groups on narrow columns). The
`Q4_K` code range `0..15` gives `|dot| ≤ 32·15·127 = 60,960`, and the affine term
`S·(d·sc·dot − dmin·m·xsum)` needs `i128` intermediates (`|dmin·m·xsum| < 2^66`), so the shared
loop's bound must admit a wide carrier (`narrow = false`, `top = 1008`) rather than reuse the
`Q5_0` narrow bound.

`Q6_K` repeats the Q4_K shape with a lighter affine term and a different scale indexing:

| dial | `gemm_q6_k` (`kcols.rs:1461`) | shared row kernel (`kcols.rs:345`) |
|---|---|---|
| block | 256 weights, 210 B: `ql[128]@0`, `qh[64]@128`, `sc[16]@192`, `d@208` | same |
| groups | 8 per block, 16 signed `i8` scales, 2 per group | same |
| code | 6-bit unsigned `0..63` minus 32 → `[−32, 31]` | same |
| group dot | scale-folded 16-lane pairs, `sc0·dot0 + sc1·dot1`, `i128` | same |
| affine | `w_min = 0`, `xsum = 0` | same |
| lane | packed 8 columns, per-column `emin`/`ke`/`spread`, `i64` shift lanes | exact `i128` fold only |
| row decode | `Q6KRow` once per row | block scales re-read per 32-column group |
| task plan | time-sliced, direct column writes, no staging cap | `n_out×k` stage, `2^22`-word cap |

A 16-lane dot is at most `16·32·127 = 65,024`; with `|sc| ≤ 127` a 32-lane group is at most
`16,516,096`, which the existing `top = 4096` covers (`32·4096·127 = 16,646,144`). The row kernel
has no `i64` block product, so its bound is the wide `i128` one (`narrow = false`), not
`gemm_q6_k`'s narrower admission.

Two shape facts decide the fold order. First, the shared loop's staging cap
`MAX_POOLED_WORDS = 2^22` (`matmat.rs:59`) refuses a 7B FFN `18,944×256 = 4.85M` and a 152k head
`×256 = 38.9M`; the packed entries have no such cap because they write columns directly. A fold that
routes these carriers through `pooled` without the direct-write task plan sends the 7B FFN and head
to the per-column oracle. Second, the head's multi-column route (`step/cols.rs:596`) admits only
`Q8_0` and `F16`; a `Q6_K` head returns `None` and `lm_head_cols` falls back to one per-column
projection per position. Decode is not affected — `k = 1` takes `lm_head_fixed_sched` →
`matvec_q8act` → `q8act_accumulate_row::<Q6_KDecoder>` — so the gap advances **Q2** (perplexity
scoring), not T1.

## 2. The target

One packed loop, entered by `matmat_q8act`, with the row kernels as its `k = 1` case:

| dial | values | where it lives now |
|---|---|---|
| activation panel | one column (`k = 1`), packed 8-column groups (`k > 1`) | `gemm.rs:104-206` |
| row decoder | `Q8_0`, `Q5_0`, `Q4_K`, `Q6_K`, ternary, packed | `FastBlockDecoder`, `Q4KRow`, `Q6KRow` |
| columns per lane | 4, 8 | `dot32x4_codes`, `group2_x2` |
| row tile | 1, 2 | `pooled_rows`, `group2_x2` |
| accumulator | checked `i128`, power-of-two `i64` lanes | `pooled_rows`, `finish` |
| task plan | interleaved `n_out×k` stage, or time-sliced `k` columns | `pooled`, `run_global` |
| bound | narrow, wide | `fast_bound_ok` |

The loop walks the weight rows once and folds each decoded group against every packed column. A
carrier supplies the row decode and the bound; the panel, lane count, row tile and task plan are
independent dials. The three `gemm_*` entries become carrier values plus dial selections; their
direct tests either move to the shared entry or delete with the entry.

## 3. Order of work

The order follows what runs on this host. `gemm_q4_k` and `gemm_q6_k` admit AVX2
(`kcols.rs:1406,1461`), so their fold is measurable here; `gemm_q8_0` needs AVX-VNNI, so its
transplant is not.

1. `Q4_K` first, as the shared loop's carrier and fallback for columns `gemm_q4_k` refuses. Bring
   the row kernel over with the wide bound and the xsum vector; keep `gemm_q4_k` until the paired
   test shows the shared arm inside its spread on this host. The carrier is in
   (`bench/cpu/1794806002-q4k-carrier.md`); the paired test puts it **5.78–9.58× slower** than
   `gemm_q4_k`, so the entry stays and the packed lane (`Q4KRow`, `q4_k_row_lanes_avx2`) is the
   next dial, not a deletion.
2. `Q6_K` second, on the same dials.
3. `Q8_0` last: the packed dial transplant (panel, 8 lanes, 2-row tile, `i64` accumulator) is for
   AVX-VNNI hosts; on this laptop the shared loop already carries the carrier, and the value tests
   are the only evidence available. Preserve both the AVX2 twin and the packed entry until a VNNI
   host times it.
4. The `Q6_K` head's multi-column arm is a separate change in `step/cols.rs`, after the carrier is
   in the shared loop; it advances Q2, not T1, and is not a reason to reorder the fold.

The direct-write task plan is a prerequisite for deleting either `gemm_q4_k` or `gemm_q6_k` from the
7B path, not a Q6_K detail. `pooled` stages an interleaved `n_out·k` buffer and refuses above
`MAX_POOLED_WORDS = 2^22` (`matmat.rs:58,416`); the 7B FFN `18,944 × 256` is 4.85M words and the
152k head `×256` is 38.9M. Routing those calls through `pooled` without the plan sends them to the
per-column oracle, which is the silent catastrophe the fold exists to avoid. The vehicle already
exists: `BandTask` (`chart/task.rs:39`), `PBand<i64>` (`chart/pool/chunks.rs:237`) and
`for_pbands` (`:280`) write disjoint column bands with no output stage, and `kern::gemm` already
uses them. The plan is to replace the interleave with `k` empty output columns, hand each task a
`PBand`, and keep `Option<Vec<Vec<i64>>>` and the refusal semantics unchanged; the only cost is the
extracted `pooled_rows` signature.

Each step keeps `kquant_columns_match_per_column_matvec_q8act` (`matmat.rs:1200`) and
`columns_match_per_column_matvec_q8act` (`:1007`) green. A mutation — dropping `− w_min·xsum`,
swapping the scale/min bytes, a wrong xsum stride, or a one-column lane order — must turn them red;
restoring it must turn them green.

## 4. The measurement that decides each step

Shapes `n_out×n_in ∈ {3584×3584, 1024×3584}`, `k ∈ {32, 128, 512}`, both routes on one binary,
three runs per cell, mean and spread. No command runs both today: `bench_q8_0_gemm_shapes`
(`gemm.rs:3172`) is gemm-only and `pooled_rate_at_fit_shape` (`matmat.rs:1486`) is the shared loop on
its own shapes. The fold adds the bypass flag only for the measurement and removes it with the
entry. The step is accepted when the shared loop is inside the spread of the entry it replaces; a
loss outside the spread is reported with the dial that causes it, not averaged away.

## 5. Not covered

- No timing in this receipt; the dial differences are read from the source. The paired measurement
  `pooled_against_gemm_q8_0` (`src/engine/kern/matmat.rs`, `--ignored`) is the first data point and
  is reported separately.
- The `Q4_K`/`Q6_K` time-sliced task plan and the `n_out·k` budget interaction are described, not
  measured.
- The AVX2 twin's arithmetic is unchanged; the fold does not touch the portable oracle.
- The head gap is scoped to multi-column scoring; the decode `k = 1` head route is unchanged.
