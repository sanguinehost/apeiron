# nq: one 1,826-column prefill pass against two 913-column passes (1.12×)

Criterion: **T2** (turn latency), partial. The subject is the cold CPU prefill of the 0.6B pack at
width 8 on the OVH box: the 1,024-column default splits a 1,826-column prompt into two 913-column
passes, and this receipt measures that split against one 1,826-column pass, changes the default block
to 2,048, and records the per-dispatch `pool.wait` distribution that locates the per-pass cost. It
does not close T2: the laptop, the 7B and the outside-pass allocation/insert (the area-1 lane) are
not measured here. On the merged tree (this change plus the area-1 parallel snapshot clone) the same
cold turn is **4.078 s** against llama's **3.958 s**, 1.030× (`## The merged tree`).

Grades: `measured` for every wall, phase mean, wait, class count, percentile bound, RSS and the
traced-vs-untraced check; `model` for the 115-column row-pass count, the attention pack count, the
`pass − wait` comparisons (llama's own barrier wait is unmeasured), the 196 NQ-linears and
84 = 3 × 28 class mappings, the scheduling-noise attribution and the share of the llama gap closed.
`tested` for the histogram path's unit and integration tests (gate section). Nothing here is
`proved`. Raw lines: `1795683267-nq-prefill-one-pass.log`.

Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan, 8 sockets × 1 core × 1 thread,
31.3 GiB, 256 MiB L3, AVX2 and FMA, no AVX-512, no VNNI), no GPU. The laptop — the only device a
timed claim may name — is **not measured**. Times UTC; every job under `~/bin/adg run --` (16 GiB
cap, 8 jobs, swap zero), 8 participants (`APEIRON_THREADS=8`), cache 440 MiB.

**Base.** Repo HEAD `1a4df32f` plus the change below and the `pool.wait` histogram. The binary is the
gate rebuild on the box from the synced final tree (`cargo build --release --locked --example
prefix_solve`, 09:46:29Z), sha256
`a63b72208f1936758b9a44be999665f2ee60bde2adaf68623a1560a65015599d`; the 09:52:20Z invocation at
the start of the timed suite recompiled nothing (`Finished … in 0.02s`) and the copied binary has
the same sha256. Pack `nq06-v3.nqp`, skeleton
`skeleton-q8head.gguf`, llama `Qwen3-0.6B-Base.f16.gguf`, `llama-bench` from llama.cpp `049326a00`.
Two windows: traced 09:52:20–09:55:13 (start load 0.93), untraced 09:56:39–09:57:11 (start load
1.40). Command: `prefix_solve --model … --pack … --tokenizer … --len 1827 --differ 0 --decode 0
--mb 440` plus `--phases` in the traced window, against `llama-bench -p 1827 -n 0 -t 8 -r 1`.

## The result

Six paired rounds in two windows: `1,024` is `APEIRON_PREFILL_COLS=1024` (2 × 913 columns plus a
one-column tail), `2,048` is the new default (`ceil(1826/2048) = 1`, one 1,826-column pass plus the
same tail). The 1,827th token is the one-column tail in both arms. Every round runs 2,048 then 1,024,
except round 2 of each window (reversed). `spread` is `(max − min) / min`.

Traced window, llama interleaved:

| | run 1 | run 2 | run 3 | mean | median | spread |
|---|---:|---:|---:|---:|---:|---:|
| llama `pp1827`, t/s | 462.93 | 463.37 | 462.55 | 462.95 | 462.93 | 0.18% |
| llama, s | 3.947 | 3.943 | 3.950 | 3.946 | 3.947 | 0.18% |
| 1,024 (2 × 913), s | 4.790 | 4.788 | 4.716 | 4.765 | 4.788 | 1.57% |
| 2,048 (1 pass), s | 4.266 | 4.299 | 4.249 | 4.271 | 4.266 | 1.18% |
| 1,024 / 2,048 | 1.123 | 1.114 | 1.110 | **1.116** | **1.114** | 1.16% |

Untraced window, same binary and workload:

| | run 1 | run 2 | run 3 | mean | median | spread |
|---|---:|---:|---:|---:|---:|---:|
| 1,024 (2 × 913), s | 4.810 | 4.796 | 4.799 | 4.802 | 4.799 | 0.29% |
| 2,048 (1 pass), s | 4.264 | 4.265 | 4.327 | 4.285 | 4.265 | 1.48% |
| 1,024 / 2,048 | 1.128 | 1.125 | 1.109 | **1.121** | **1.125** | 1.71% |

The 2,048 arm wins **all six rounds**; the factor is 1.109–1.128, pooled mean **1.118** and median
**1.118** over the six paired ratios. The traced window is the one with llama: there 2,048 is 1.082×
llama's `pp1827` and 1,024 is 1.207×, so the change closes 60.3% of the gap (`model` on the means).
The untraced window has no llama run; its 2,048 mean crossed with the adjacent traced llama is
1.086×, but that crosses windows and is not used. The two windows differ by 0.33% (2,048) and 0.78%
(1,024) at the means.

Trace check, default arm, same configuration: per-round traced − untraced = **−1, +67, −29 ms**
(median **−1 ms, −0.02%**; mean +12 ms, +0.29%; range −0.68% to +1.58%). The sign is not stable and
every difference is inside the arm spread, so no trace correction is applied (`model`); the earlier
+47 ms attribution from a prior binary is dropped. The 1,024 arm has no untraced twin in the suite,
so this check does not cover it.

## The phase split (traced window, mean of 3)

`head` runs outside `Phase::Pass` (the greedy step after the pass) and is shown but excluded from
the leaf sum. The untraced remainder is `pass − Σ leaves`.

| term | 1,024 (2 × 913) | 2,048 (1 pass) | delta |
|---|---:|---:|---:|
| cold wall, ms | 4,764.7 | 4,271.3 | **−493.3** |
| `embed` | 9.998 | 6.127 | −3.871 |
| `norm` | 175.374 | 133.144 | −42.231 |
| `residual` | 132.455 | 64.426 | −68.029 |
| `attn.proj_in` | 614.576 | 585.142 | −29.434 |
| `attn.core` | 1,525.015 | 1,421.088 | −103.926 |
| `attn.proj_out` | 248.914 | 239.843 | −9.071 |
| `ffn` | 1,636.780 | 1,440.794 | **−195.986** |
| leaves, sum | 4,343.111 | 3,890.564 | −452.547 |
| untraced (`pass − leaves`) | 45.006 | 34.734 | −10.272 |
| `pass` | 4,388.117 | 3,925.299 | **−462.819** |
| `head` (outside `pass`) | 5.874 | 6.278 | +0.404 |
| `pool.wait` (overlay) | 331.201 | 210.666 | **−120.535** |
| `pass` calls | 3 | 2 | −1 |
| `pool.wait` calls | 1,295 | 902 | −393 (−30.3%) |

The wall delta (−493.3) is 30.5 ms larger than the pass delta (−462.8); the outside-pass work (one
`decode_state`, one cache insert, probes, the head) is the same set of operations in both arms and
its arm means differ by 30.5 ms (`model`: the per-run wall − pass is not recorded). `pool.wait` is
an overlay nested inside the leaves, so it is not added to them.

`pass − wait` is computed per run and then reduced: 1,024 = 4,087.677 / 4,054.385 / 4,028.687 ms,
median **4,054.385**; 2,048 = 3,732.652 / 3,699.750 / 3,711.497 ms, median **3,711.497**; delta
**−342.888 ms** on the per-run medians (mean delta −342.283). It is an upper bound on the wait-free
pass (any wait the histogram does not see remains), not a decomposition: llama's own barrier wait
is unmeasured and the caller's outside-pass waits are not in either pass.

## Where the 463 ms is

The leaves decompose the pass delta exactly: −452.547 ms of leaf wall and −10.272 ms of untraced
work. The largest terms are `ffn` (−196.0), `attn.core` (−103.9), `residual` (−68.0), `norm`,
(−42.2) and `attn.proj_in` (−29.4). The removed pass also removes 393 of 1,295 dispatches and
120.5 ms of caller wait (26.0% of the pass delta for 30.3% of the calls), and the wait sits inside
the leaf wall, so 332.0 ms is an upper bound on the work part of the leaf drop and the rest is
measured wait (`model` on the nesting).

The per-pass work has two candidates and this measurement does not separate them: the second pass
walks every weight tensor again for its 913 columns, and it pays the per-call pool setup and barrier
again. The 0.6B pack is small (0.977 b/w, ~73 MB), so a second weight stream at the box's bandwidth
is a few milliseconds, not the 196 ms `ffn` shows; the bulk of the leaf drop is per-call cost whose
operator split is not located. The direction is per pass, not per column: the column count is
identical in the two arms.

## The attention core is not the lever

The score and softmax counts depend only on the query positions, not on the pass boundary. On this
host `attn_core_tile` takes the AVX2 body (`attn_core_tile_avx2_inner`, `src/engine/attn/f32_core.rs:1416`):
`gemm_tile` at line 1563 scores a full `KV_TILE = 64` column block for every active row, so the
1,826 positions cost `Σ (n − at0)·64 = 1,726,080` dots per query head, and the causal softmax prefix
`end = kv_tile.min(cut + 1 − at0)` at line 1574 reads `n(n+1)/2 = 1,668,051`. Both are equal for
2 × 913 and 1 × 1,826 (`model`: counted from the loop over `pos`).

The KV pack is not invariant: the pack loop starts at position 0 for every query tile
(`QUERY_TILE = 64`, `src/engine/step/softmax.rs:40`), so the split's second pass re-packs the first
913 keys — 435 KV-tile packs for one pass against 464 for 2 × 913, per layer and KV head (+6.7%,
`model`). Measured `attn.core` moves **−103.9 ms (−6.8%)** at the means, in the one-pass arm's
favor. The pack count alone is too small to account for all of it; the +503 ms square-score model
from the earlier draft is contradicted and withdrawn, and a 1.33× attention penalty is excluded.

## The pool wait is broad, not one class

`run_dyn` opens the `PoolWait` span after the caller's own partition and records the dispatch's task
count `n` in the span's `cols` (`src/engine/chart/pool.rs:592`). The histogram buckets each wait by
`floor(log2 ns)` in eight task-count classes. A printed bound is the bucket's upper edge, so
`p50 < 131072 ns` means the median is in `[65536, 131072)`.

The class counts are identical in all three runs; the bucket edges move more than in the earlier
pair, and one split run carries two large maxima (14.0 ms in `5-8`, 29.4 ms in `65+`). The table
gives the calls and the p50/p95/max range over the three runs.

1,024, cold (`pool.wait` 331.201 ms mean, 1,295 calls):

| class | calls | p50, 3 runs | p95, 3 runs | max, 3 runs |
|---|---:|---|---|---:|
| `1`–`4` | 0 | | | |
| `5-8` | 450 | 131.1 µs | 1.049 ms | 2.540–13.996 ms |
| `9-16` | 173 | 524.3 µs | 1.049 ms | 0.900–2.087 ms |
| `17-32` | 504 | 4.1 µs | 1.049 ms | 0.781–0.964 ms |
| `65+` | 168 | 524.3–1048.6 µs | 1.049–2.097 ms | 2.037–29.431 ms |

2,048, cold (`pool.wait` 210.666 ms mean, 902 calls):

| class | calls | p50, 3 runs | p95, 3 runs | max, 3 runs |
|---|---:|---|---|---:|
| `1`–`4` | 0 | | | |
| `5-8` | 254 | 262.1 µs | 2.097 ms | 4.999–5.748 ms |
| `9-16` | 116 | 8.2 µs | 0.524–1.049 ms | 0.723–3.061 ms |
| `17-32` | 448 | 4.1–8.2 µs | 0.524–1.049 ms | 0.936–1.327 ms |
| `65+` | 84 | 262.1–524.3 µs | 2.097 ms | 1.722–3.871 ms |

The count differences are measured: the removed pass contains 196 `5-8` calls, 57 `9-16`, 56
`17-32` and 84 `65+`. The mapping of the 196 to the pack's 196 NQ linears, one dispatch each because
`project_pooled_body` dispatches one column band per participant (`width = 8`,
`src/engine/kern/nq.rs`), and of the 84 to 3 × 28 layers is `model`; the mapping beyond those is not
established. No class owns the wait: the per-class p50 spans 4.1 µs (`17-32`) to 1048.6 µs
(`65+`), the `9-16` median falls 524.3 → 8.2 µs when the split goes, and the total falls 36.4% for
30.3% fewer calls.
The cost is per dispatch and per pass, not one straggler; the two large maxima are single-run
scheduling noise, not a change in the class medians (`model`).

## The band was not the lever

At 913 columns the NQ band is 120 columns: `ceil(913/8) = 115`, rounded up to eight for the wide
tile, so seven tasks take 120 and the last takes 73. A 115-column band leaves 3 leftover columns in
each full band; the leftover loop folds a pair and then a single (`dot_rows2`/`dot_rows`), so a full
band costs 2 row passes there — 14 + 2 = **16 row passes against 15** for the group-aligned 120. The
earlier ≤102.55 ms loose band bound was a mirage; the group-aligned band was already the cheaper
shape. Grade `model`: counted from the loop, not timed (`1795674629-nq-prefill-serial.md`). With the
2,048 default the band is 232 columns and this row-pass question moves with it; it is not measured.

## Width 4 and 3,000 columns

Both are untraced pairs on the same binary; 3,000 columns is 2,999 prefill columns, and the splitter
is near-equal (`prefill_cols_width`, `src/engine/step.rs`), so 2,048 takes 2 passes (1,500 + 1,499)
against 3 for 1,024 (1,000 + 1,000 + 999); the 3,000-column runs reserve a
700 MiB cache, the 1,827-column pairs 440 MiB. The 3,000-column row is a separate
three-round pair at 09:58:09Z, because the pair run at the end of the traced suite had one round at
1.004× under rising load.

| workload | 1,024, s | 2,048, s | 1,024 / 2,048 | per-round |
|---|---:|---:|---:|---|
| width 4, 1,827 tokens | 7.240 | 6.869 | **1.054** | 1.053, 1.056, 1.053 |
| width 8, 3,000 tokens | 8.108 | 7.835 | **1.035** | 1.038, 1.025, 1.041 |

Width 4 wins all three rounds, so the >= 4 threshold stays; the measured gain is 5.4% against 11.6%
at width 8 in the traced window. At 3,000 columns the gain is 3.5% and every round wins
(1.025–1.041), so the default is not a 2,048-and-under artefact; it is one prompt on one host, and
prompts past 3,000 columns are unmeasured. The width-4 claim is the cold prefill only; it is not a
phase split. The 1.054 is the paired mean; the ratio of the medians is 7.237 / 6.876 = 1.0525,
which is the statistic the `prefill_block` doc cites (`model`).

Median peak RSS over the three cold runs, `/usr/bin/time -v`, width 8, 1,827 tokens: 1,362,544 KiB
(1.299 GiB, 1,024) against 1,429,424 KiB (1.363 GiB, 2,048), **+66,880 KiB = +65.3 MiB (+4.91%)**.
The delta is the wider pass's column buffers (`model`); the absolute includes the 440 MiB cache
reservation and the outside-pass state, so it is not a C1 figure. The earlier receipt's 1,294,188 KiB row was a
different harness invocation (`--differ 205`) and is not comparable; this pair is measured the same
way in both arms.

## The change

`prefill_block()` defaults to 2,048 on a pool of four or more participants, 256 below
(`src/engine/step.rs`). On this prompt `ceil(1826/2048) = 1`, the measured arm exactly; the override
`APEIRON_PREFILL_COLS` is unchanged. The doc comment carries the pair by pointer. This makes the
measured arm the default; it does not re-measure a 2,048-column prompt, the 7B, or the laptop.

## The merged tree

The two lanes are separate binaries: every number above is `a63b7220`, area 2 alone. The merged tree
is area-2 `0b61fcbb` plus the area-1 cherry-pick `678b9e97` (parallel snapshot clone), rebuilt on the
box at 10:12:47Z to sha256
`d1b02e32452d19fcc69c604e247d5a10f1001de602fed94d38ee60196937f60a`. Only markdown changed after
that build. Workload, host, width and cache are the same as the untraced window; `--phases` is off.
llama is interleaved.

| | run 1 | run 2 | run 3 | mean | median | spread |
|---|---:|---:|---:|---:|---:|---:|
| llama `pp1827`, t/s | 459.53 | 460.87 | 464.52 | 461.64 | 460.87 | 1.09% |
| llama, s | 3.976 | 3.964 | 3.933 | 3.958 | 3.964 | 1.09% |
| merged cold, s | 4.158 | 3.990 | 4.085 | **4.078** | **4.085** | 4.21% |
| merged / llama | 1.046 | 1.007 | 1.039 | **1.030** | **1.039** | 3.90% |

The merged turn is 1.047× the area-2 2,048 mean (4.271 → 4.078 s, −193.7 ms) and 1.169× the 1,024
arm (4.765 s) (`model`: ratios of the measured means); it closes 85.1% of the 1,024-to-llama gap
(`model` on the means) and remains **3.0% (120 ms) slower than llama**. The remaining outside-pass `decode_state` allocation, 207 ms on the
area-1 binary, is 5.1% of this turn and is untouched; a single-term ceiling there is
`1/(1 − 207/4078) = 1.054×` (`model`: the alloc is not measured on the merged binary, and removing all
of it is not claimed). Peak RSS median over the three merged runs is 1,532,916 KiB = 1.462 GiB,
**+103,492 KiB = +101.1 MiB (+7.24%)** over the area-2 2,048 row; the delta is not isolated (the
parallel clone's worker stacks and allocator arenas are candidates), and it is not a C1 figure.
The merged tree passes the contributor gate with the three known `glslc` failures: the build
reproduces `d1b02e32…`, fmt and clippy `--all-features` exit 0, lib **695 passed, 1 failed,
55 ignored**, `spirv_storage_race` 2 passed and 2 failed, and every doc check exits 0 (the `.log`,
merged gate section).

## The gates

On the final tree on the OVH box, under `~/bin/adg run --` (summary and failing blocks in the
`.log`):

| command | result |
|---|---|
| `cargo build --release --locked --example prefix_solve` | rebuilt from the final tree at 09:46:29Z, sha256 `a63b7220…`; the timed copy has the same hash |
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --workspace --all-features --all-targets --locked -j 1 -- -D warnings` | exit 0 |
| `cargo test --workspace --locked --no-fail-fast` | lib **694 passed, 1 failed, 55 ignored**; `spirv_storage_race` 2 passed, 2 failed; `trace_wait_reset` 1 passed; every other suite passed |
| `python3 tools/check-claims.py` | exit 0 |
| `python3 tools/book-check.py` | exit 0 |
| `mdbook build book` | exit 0 |
| `mdbook test book -L $HOME/apeiron-mdbook/debug/deps` | exit 0, every chapter |

The three failures are the GPU-compiler pin, not logic:
`gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl` (glslc exit 1: `#extension : extension not
supported: GL_EXT_integer_dot_product`; `src/gpu/spirv/coopmat_gemm.comp:9`),
`attn_coopmat_spirv_is_the_glsl` (glslc SIGSEGV) and `splitk_reduce_spirv_is_the_glsl` (the committed
`.spv` is stale against the box's shaderc 2023.8). `cargo test --all-features` needs the `glslc` pin
and a device and is not run on either host.

The histogram tests: `waits_bucket_by_floor_log2` and `dispatch_sizes_class_by_width` cover the
helpers; `a_pool_wait_span_reaches_the_histogram` drops a real `Phase::PoolWait` span and checks
`pool_wait_histogram()` and `snapshot()`; `reset_zeroes_the_wait_table` is the
`tests/trace_wait_reset.rs` integration target, so its process has no other span writer, and it
asserts exact emptiness after `reset` — deleting the `clear_wait_table` call from `reset` turns it
red. The lib test shares the process-global wait table with the `#[ignore]`d `gemm_probe` and
`gemm_attrib` instruments, which call `reset` and `set_enabled`; the default harness runs neither,
so its pass is evidence for the default gate only, and an `--ignored` run can race it.

The maintainer gate (pinned Aeneas, Charon and the corpus Lean build) is not bootstrapped on either
host, and the committed extraction is already stale: `proof/ApeironWord.lean` annotates `trace.Span`
at `trace.rs:139-145`, `trace.span` at `151-163` and `step.prefill_block` at `step.rs:385-394`, while
this tree has them at 240, 252 and 478. `trace.Span`/`trace.span` and `step.prefill_block` are opaque axioms whose
signatures are unchanged; the histogram and the default value are outside the extracted bodies, so
regenerating would move annotations, not change a body. This change moves all three annotations
further, so `proof/regen.sh --check` fails until the extraction is regenerated.

## What this does not cover

- The laptop (the target device) and the 7B pack are unmeasured; the timed host is the box.
- Width 8 is the traced pair; width 4 and 3,000 columns are untraced wall pairs with no phase split.
  Widths 2–3, 5–7, and prompts past 3,000 columns are unmeasured; a 2,048-column prompt itself is
  not timed (the 1,826 and 2,999-column prompts are), and the 3,000-column case is one prompt.
- The outside-pass interval (`decode_state` + `insert`) is the area-1 lane and is not changed
  here; that lane measures ~421 ms at 1,024 on its instrumentation-only binary
  (`1795683289-nq-cold-outside.md`). In this tree wall − pass is 376.6 ms and wall − leaves 421.6 ms,
  the difference being the 45.0 ms untraced pass.
- The class-to-operator mapping beyond the 196 NQ linears is not established.
- Rustc/LLVM code generation, the kernel ABI, Vulkan drivers and the `unsafe` container are not
  covered, as usual.

## Review

Independent review: `gemini-4-hydrogen-xhigh` (xhigh), read-only, on an earlier receipt and tree;
verdict **FAIL, 11 refusals** (one-path and `unsafe` passed). The refusals and their disposition:

| # | refusal | disposition |
|---|---|---|
| 1 | 1.077 was `median(default) / median(one pass)`, not the paired-ratio median | paired means and medians are separate rows in both windows; 1.077 is not used |
| 2 | `head` is outside `pass` but was in the leaf sum | `head` is outside the leaf sum; untraced is `pass − leaves` (45.006 → 34.734) |
| 3 | `pass − wait` subtracted across runs | per-run `pass − wait`, then the median: 4,054.385 vs 3,711.497, −342.888 |
| 4 | +47 ms "trace overhead" was an unpaired prior-binary drift | measured on the final binary: −1/+67/−29 ms, median −1 ms (−0.02%); attribution dropped |
| 5 | AVX2 path, not the scalar twin, runs on the box | cites `attn_core_tile_avx2_inner` (`f32_core.rs:1416,1563,1574`) with both counts and their invariance |
| 6 | p50 summary wrong for the `9-16` class | the table gives the 3-run ranges; `9-16` is 524.3 µs split, 8.2 µs one pass |
| 7 | `ffn` called flat; `residual` called serial | `ffn` −196.0 is reported; `residual` is the pooled column add (`step::cols::add_cols`), and the book now says the add is serial within a column |
| 8 | timed binary predates the 2,048 edit; script `BIN` mismatch | final binary rebuilt after the edit, sha `a63b7220…`; the script and log carry the same name |
| 9 | 2,048 for all widths >= 4 without width-4/RSS/2,048+ data | width 4 (1.054 paired mean; 1.0525 med/med), 3,000 columns (1.035) and RSS (+65.3 MiB) measured; the threshold stays at 4 |
| 10 | histogram tests covered only the pure helpers | real `PoolWait` span test plus the `trace_wait_reset` integration target, which asserts `reset`'s own clear |
| 11 | contributor gate incomplete; no gate output in `.log` | `check-claims`, `book-check`, `mdbook build` and `mdbook test` run; the summary and failing blocks are in the `.log`; maintainer gate and its stale extraction recorded |

Second review: a fresh `general` subagent (DeepSeek Flash harness; the external `mimo`, `gemini` and
`space-bunny` reviewers were unavailable), read-only, on this receipt, diff and tree; verdict
**FAIL, 8 refusals**. All eight are fixed above and in the `.log`:

| # | refusal | disposition |
|---|---|---|
| 1 | trace-check median was run 3's −29 ms | median is −1 ms (−0.02%); the range and mean are stated |
| 2 | pairing order inverted | rounds run 2,048 then 1,024, except round 2 |
| 3 | 3,000-column passes are near-equal, not block-width | 1,500 + 1,499 against 1,000 + 1,000 + 999 |
| 4 | ~421 ms is wall − leaves, not wall − pass | outside-pass is 376.6 ms here; 421.6 ms is wall − leaves |
| 5 | `pass − wait` called a lower bound | it is an upper bound on the wait-free pass |
| 6 | `step.rs` says 1.052 against the receipt's 1.054 | 1.0525 is the ratio of medians, now named in the width-4 section |
| 7 | first glslc cause was absent from the `.log` | the `coopmat_gemm.comp` stderr lines are in the `.log` |
| 8 | class-to-operator mappings not graded `model` | the 196 and 3 × 28 mappings and the noise attribution are `model` |

Its non-blocking notes are dispositioned: the 30.5 ms drift and the trace correction are `model`; the
lib test's `--ignored` race is recorded in the gate section; the book's "residual add stays serial"
is now "serial within a column"; the maintainer note says "not bootstrapped"; the script bodies and
fixture paths are in the `.log`; and the 09:52 build is recorded as a no-op over the 09:46 binary.
The `step.rs` attention shorthand is not changed (a source edit would change the timed binary); the
receipt's attention section gives the full count.

The review is critique, not evidence: every disposition resolves to a table above or a line in the
`.log`.
