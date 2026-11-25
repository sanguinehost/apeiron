# nq: one activation scale per 1,024 elements

Criterion: **T2** (turn latency), partial: the cold 1,827-position CPU prefill of the 0.6B pack. The
change also moves the activation quantizer, so its quality screen is reported here; it is not a
context-quality (C2) claim. It is not a 7B result and not a 200k-context turn.

Grades: `measured` for the walls, the PPL pair, the phase split, the profile and the microbenchmark;
`tested` for the grouped kernel equality and the quantizer contract. The grouped equality is
`tested` against the per-column twin, not `proved`: it lives in an `unsafe` AVX2 body the extraction
does not model. Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan, 8 sockets × 1
core × 1 thread, 31.3 GiB, 256 MiB L3, 2.4 GHz base, no AVX-512, no VNNI), no GPU. The laptop — the
only device a timed claim may name — is **not measured** (Omissions). Times UTC.

Model: `~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf`; pack
`~/models/nq-scratch/nq06-v3.nqp` (196 linears, 28 layers, 0.977441 b/w); harness
`examples/prefix_solve.rs`, `--len 1827 --differ 205 --decode 0 --mb 440`, width 8,
`APEIRON_PREFILL_COLS=1024`. Every timed and test run under `~/bin/adg run --` (16 GiB cap, 8 jobs).

## What changed

`src/engine/kern/nq.rs`, `src/engine/chart/mod.rs`, `src/engine/chart/tests.rs`.

- `chart::quant_act_grouped(x, group)` is the quantizer: one `act_scale` of the group's largest
  magnitude, repeated on every 32-wide block the group covers, and each block's codes rounded
  against it. `chart::quant_act(x)` is the wrapper `quant_act_grouped(x, 1)` and is bit-identical to
  the previous per-block function; the chart FFN and head path (`quant_act_pooled`) still calls it
  with group 1.
- `kern::nq::nq_act` is the one NQ call site, `quant_act_grouped(x, NQ_ACT_GROUP)` with
  `NQ_ACT_GROUP = 32` blocks (1,024 elements); `t`, `z`, the residual `x`, `quantized_band` and
  `quantized_slices` call it.
- `CodeTerms::uniform_group` detects aligned runs of equal block scales, `CodeTerms.group` and
  `ColRef.group` carry the run, and `code_select_row8_avx2` folds a run behind one `i16` pair
  accumulator and one `vpmaddwd`/`vpsllvd`/`vpaddd` instead of one per block. The run is taken only
  when all eight columns report the same group; otherwise the same kernel takes its per-block loop,
  which is the previous arithmetic. `code_select_avx2`, `code_select_block2_avx2` and the block
  dispatch keep per-block scales and are exact for either shape.

There is no new projection function: the change is a dial inside the one row loop. The bound the run
relies on is `2·127·G ≤ 32767` in the `i16` pair (G ≤ 129); `NQ_GROUP_MAX = 32` and
`uniform_group` refuses a longer or unaligned run. The shift vectors are equal across a run because
they are derived from the scale, so the run's first vector is the run's. The `i32` lane accumulator
sees the same value the per-block chain produces, so `CodeTerms::new`'s existing `i32` and `i64`
admissions still cover it.

The end-to-end test `the_grouped_scale_reaches_the_pooled_reader` runs `NQ_TILE_COLS × pool width`
columns: `project_pooled_body` gives each participant `ceil(k/width)` columns and runs the wide tile
only when a band holds eight, so a nine-column fixture reaches only the per-column fallback on a
multi-participant pool. The test also asserts that `nq_act`'s scales are the grouped quantizer's at
`NQ_ACT_GROUP`, so reverting the call site fails it.

The dial is a loop difference against the pin's repacked q8_0 GEMM, `ggml_gemm_q8_0_4x8_q8_0`
(`ggml/src/ggml-cpu/arch/x86/repack.cpp`, `~/llama.cpp` at `049326a00`): llama.cpp applies the block
scale inside a block GEMM, while the NQ path re-derived a shift per 32 elements. Grouping equal
scales removes 31 of every 32 shift/madd steps without changing the block layout the rest of the
reader reads. That comparison is an `anchor` read from the pin's source, not a measurement in this
receipt.

## Quality

Same harness, same stream: 8,176 tokens of `~/models/wiki.test.raw` (16 windows of ≤ 512), `--kernel
q8act --head-batch 16`, two binaries in one window on the box: `/tmp/ppl.head` (`8f6287eb…`, built
from `git show HEAD:` of the three changed files) and `~/bin/ppl.final` (`a540bc6a…`, rebuilt from
the final source; it reproduced the earlier `group2` pair's values exactly). This is a two-binary
A/B on one stream, not a same-binary control; the HEAD side is HEAD's body, not a runtime switch.

| arm | PPL | mean nll | tok/s | wall s |
|---|---:|---:|---:|---:|
| HEAD (per-block scale) | 54.797134 | 4.003638 | 158.0 | 51.8 |
| grouped (G = 32) | **53.991898** | **3.988834** | 169.5 | 48.2 |

The grouped scale is **0.985× HEAD (−1.5%)** on this stream, not worse. A design screen with a
temporary `APEIRON_ACT_QK` dial put G = 1 at 54.797134 and G = 32 at 54.515891 on the same 8,176
tokens. That dial replaced every `quant_act` caller, the chart head included, so its G = 32 row is a
different configuration from the shipped NQ-only grouping; the dial is not in the tree and no graded
row uses it. Only the pair above is graded.

The per-block chain remains available to any `QuantAct` whose scales are not aligned runs: the chart
FFN and head (`quant_act_pooled`), the `matmat`/`gemm` fixtures and every `QuantAct` built block by
block keep group 1, and the reader selects it per call. The scale change is NQ-only.

## Result

### Cold prefill: paired A/B

HEAD = `~/bin/prefix_solve.merge4` (`eae8ce74…`, the binary of
`bench/cpu/1795601655-prefill-onedispatch.md`); grouped = `~/bin/prefix_solve.final`
(`b7058658…`, rebuilt from the final source; its `.text` is `441aff38…`, identical to the earlier
`group2` build, so the test-only edit is code-neutral). Three interleaved rounds in one window;
another lane's guarded job had just released the box, so absolute walls are window-specific and the
ratio is the graded number.

| round | HEAD s | grouped s | grouped/HEAD |
|:---:|---:|---:|---:|
| 1 | 7.234 | 6.489 | 0.897 |
| 2 | 7.216 | 6.398 | 0.887 |
| 3 | 7.013 | 6.402 | 0.913 |
| mean | **7.154** | **6.430** | **0.899** |

The pairs are 0.897, 0.887 and 0.913; the grouped arm's run-to-run spread is 1.4%, the HEAD arm's
3.1%, and every pair moves the same way. The rate is **255.4 → 284.1 tok/s** at width 8. The grouped
artifact's whole-file hash was unchanged after the profile run (`b7058658…` before and after).

### Phase split

`--phases`, one cold run per arm, adjacent in the same window. These are single runs; the paired
wall above is the graded number and the split is read as shares.

| phase | HEAD ms | share | grouped ms | share | grouped/HEAD |
|---|---:|---:|---:|---:|---:|
| `ffn` | 2411.127 | 34.4% | **1927.581** | 31.0% | 0.799 |
| `attn.proj_in` | 1065.667 | 15.2% | **872.034** | 14.0% | 0.818 |
| `attn.proj_out` | 463.761 | 6.6% | **337.964** | 5.4% | 0.729 |
| `attn.core` | 1709.134 | 24.4% | 1736.754 | 28.0% | 1.016 |
| `norm` | 339.421 | 4.8% | 314.989 | 5.1% | 0.928 |
| `residual` | 588.863 | 8.4% | 594.996 | 9.6% | 1.010 |
| `pool.wait` | 174.017 | 2.5% | 145.062 | 2.3% | 0.834 |
| `pass` | 6615.564 | 94.4% | 5819.771 | 93.7% | 0.880 |
| cold wall | 7.005 s | | 6.208 s | | 0.886 |

The three projection leaves — the callers of the grouped linears — fall **3,940.6 → 3,137.6 ms
(0.796)**, and the `pass` leaf falls **6,615.6 → 5,819.8 ms (0.880)**; the wall falls 7.005 → 6.208 s
(0.886). The leaves fall 803 ms while the wall falls 797 ms; `attn.core` rises 28 ms (1.016) on a
body this change does not edit, and `norm`/`residual`/`pool.wait` move with it. `ffn` remains the
largest leaf (31.0%); `attn.core` is second at 28.0%. A single run per arm cannot apportion the
unchanged leaves, so none of them is attributed.

The projection leaves were **56.3%** of the HEAD wall and are **50.5%** after, so the ceiling a
further projection change can reach is **`1/(1 − 0.505) = 2.02×`** (2.29× before). The grouped
kernel removes 20.4% of the projection leaf time; the remainder is the non-dot work below.

The performance-gate instrument is `prefix_solve --phases`, not `examples/phase_split --target`:
that tool traces a decode token, and the wall this change moves is the prefill. The target group is
the three projection leaves at 56.3% of the HEAD cold wall; a further projection change inherits
`2.29×` on HEAD and `2.02×` after. Decode is measured separately below; `attn.core` is the decode
dominant term, and this change edits no attention body.

### Turn and decode

`--len 1827 --differ 205 --decode 64`, three interleaved pairs in the same window. The prefill
column is a different set of runs from the `--decode 0` pair above, so its mean is not the cold-pair
mean.

| round | HEAD prefill s | grouped prefill s | HEAD 64 s | grouped 64 s |
|:---:|---:|---:|---:|---:|
| 1 | 7.097 | 6.571 | 1.998 | 1.891 |
| 2 | 7.085 | 6.412 | 1.960 | 2.005 |
| 3 | 7.092 | 6.512 | 1.950 | 1.910 |
| mean | **7.091** | **6.498** | **1.969** | **1.935** |

The prefill half is 0.916×, the 64 decoded tokens are 0.983× (**32.5 → 33.1 tok/s**), and the turn's
total is **9.060 → 8.434 s (0.931×)**. The prefill pairs are 0.926, 0.905 and 0.918; the decode
pairs are 0.946, 1.023 and 0.979, so the decode difference is inside the run-to-run spread and is
not claimed as a decode gain. The turn's prefill mean is lower than the cold-pair mean because the
two sets of runs sit at different points in the window.

### Where the projection time is

`perf record -F 499 -g` on the grouped binary, one cold run (6.583 s), `perf report --sort symbol`
(`/tmp/opencode/group-final2.log`, children/self):

| symbol | share |
|---|---:|
| pool frame | 23.64% children, 23.38% self |
| `StageCols::dot_row` | 20.58% children, 1.52% self |
| `code_select_row8_avx2` | 19.03% children, 18.88% self |
| `gemm_tile` | 9.05% children, 8.96% self |
| `attn_core_tile_avx2_inner` | 8.97% children, 8.87% self |
| `project_row` | 7.77% children, 7.71% self |
| `ColPipeline::run` | 5.25% children, 5.14% self |
| `quant_act_grouped` | 5.07% children, 4.88% self |
| `swiglu_f32_avx2_8_into` | 2.67% |
| `rms_norm_fixed` | 2.36% |

The same kernel was 27.0% of the wall on the previous receipt's profile
(`1795601655-prefill-onedispatch.md`, shipped prefill 7.094 s); here its share is **18.9% at 6.583 s**.
`project_row` (0.51 s), `ColPipeline::run` (0.35 s) and the quantizer (0.33 s) are the next
projection terms, and the pool frame (1.56 s) is the dispatch structure the previous receipt
measured.

### The kernel in isolation

`/tmp/opencode/nqdot3.rs`: eight columns, AVX2, the same selector expansion, pair sums and shift,
verified against a scalar selection for G ∈ {1, 4, 32}; minimum of three reps, box, one core. The
grouped arm is the shape the reader now takes; A is the shipped per-block chain.

| inn × rank | A per-block | G = 32 | ratio |
|---|---:|---:|---:|
| 1024 × 640 | 66.1 | 95.2 | 1.44 |
| 2048 × 640 | 64.8 | 90.1 | 1.39 |
| 3072 × 480 | 57.3 | 101.7 | 1.78 |
| 1024 × 736 | 66.1 | 96.3 | 1.46 |
| 3072 × 736 | 57.0 | 101.3 | 1.78 |

G terms/s. The 1.39–1.78× range is the dot only; the measured projection leaves fall 0.796 in
aggregate (0.73–0.82 by leaf), which is what a dot at 45–55% of the projection leaf predicts. G = 4
was 65.7–73.3 G terms/s (1.10–1.15×) and was not taken.

### Position against llama.cpp

llama.cpp was not re-run in this window; its width-8 row is **4.00 s (3.94–4.06)** from the same box
and pin in `1795601655-prefill-onedispatch.md`. The old binary in this window is 7.154 s where the
same shipped source measured 6.957 s in that window, a window factor of 1.028; scaling the grouped
wall by it gives **6.25 s clean-equivalent**.

| arm | wall s | against llama 4.00 s |
|---|---:|---:|
| HEAD shipped (earlier window) | 6.957 | 1.74× behind |
| HEAD (`merge4`, this window) | 7.154 | 1.79× behind |
| grouped (`final`, this window) | 6.430 | **1.61× behind** |
| grouped clean-equivalent | 6.25 | 1.56× behind |

The graded comparison is **1.61×** (`6.430` over llama's `4.00 s` row); the two walls are from
different windows, and the clean-equivalent `6.25 s` applies this window's factor to the grouped arm,
giving `1.56×`. Propagating the grouped arm's three walls (6.398–6.489) and llama's 3.94–4.06 s row
gives **1.58–1.65×**, so the statement is "about 1.6× behind", not a measured interval.

The target is **1×–0.5×** (4.00–2.00 s). **Neither is reached**: a 10% wall gain where the wall must
still fall a further 1.61× for parity. The non-projection floor is `6.208 − 3.138 = 3.07 s` in the
split run (2.91 s on the quiet box of `1795601655-prefill-onedispatch.md`); the clean-equivalent floor
is `6.25 − 3.14 = 3.11 s`. With the floor held, parity needs the projection leaves to fall from
3.14 s to about 0.93 s, a further 3.4×. Half of llama (2.00 s) is 1.07 s below the 3.07 s floor and
stays refused unless `attn.core` and the other non-projection leaves give up at least that much too.

The next lever named in this receipt is `ffn` (1.93 s) plus `attn.proj_out` (0.34 s): 2.27 s of the
6.43 s wall, share 35.3%, ceiling `1/(1 − 0.353) = 1.55×`. Halving both leaves 5.30 s (1.33×
llama), so that pair alone cannot close the gap; parity also needs `attn.core` (1.74 s, 28.0%).

## Gates and falsifiers

Contributor gate on the box copy of the final source (`chart/mod.rs` `87d1276a…`, `chart/tests.rs`
`53860ffa…`, `kern/nq.rs` `c2781526…`), no GPU and no `glslc` pin
(`/tmp/opencode/group-gate4.log`):

| gate | result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | exit 0 |
| `cargo test --release --locked --lib kern::nq` | 24 passed, 0 failed, 4 ignored |
| `cargo test --release --locked --lib chart::tests` | 19 passed, 0 failed, 4 ignored |
| `cargo test --workspace --locked` | 679 passed, 1 failed: `gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl`, which needs the `glslc` pin this box does not have; 54 ignored. This is the default-feature run, not `--all-features` |
| `mdbook build book`, `mdbook test book -L $HOME/apeiron-mdbook/debug/deps` | pass |
| `python3 tools/check-claims.py`, `python3 tools/book-check.py` | pass |

The maintainer gate was **not run**: neither host has the pinned Aeneas, Charon and Lean toolchain.
It is **required**, not exempt: `quant_act_grouped` is a new body in `src/engine/chart/mod.rs`, the
`word_engine_prefill_state` closure reaches `kern::nq` (`proof/extract/lib.rs` pulls
`src/engine/kern/mod.rs`, which declares `nq`), and `proof/regen.sh --check` would compare a new
extraction against the committed `proof/ApeironWord.lean`. The move is not free inside Lean either:
the committed extraction defines `chart.quant_act_loop0*` and `chart.quant_act`
(`proof/ApeironWord.lean:8207-8372`) from the old body, and `proof/corpus/Serve/Block/Ffn3.lean`,
`Ffn4.lean`, `Serve/Block/R4/Ffn5.lean`–`Ffn9.lean` and `AxiomAudit/Serve.lean` unfold or audit those
names. The grouped body is a three-level nest (`block` → `part` → `lane`), so the post-regen loop
names are not a rename of the old `loop0_loop0`/`loop0_loop1`. After regen, `chart.quant_act` is a
one-line call and the loops are `chart.quant_act_grouped_loop0*`, so those files need a bridge lemma
for `quant_act = quant_act_grouped(x, 1)` or a re-proof. The route is `proof/bootstrap.sh`,
`proof/regen.sh --check`, then `cd proof/corpus && lake build`. The gate is an open blocker on this
change, and the bridge work is part of it, not a formality.

Review: Hydrogen (`gemini-4-hydrogen-high`) ran twice. The first pass
(`/tmp/opencode/review-group.log`) accepted the arithmetic, the `QuantAct` contract and the one-path
shape, and refused the change on the unrun gates, a test domain that missed the
grouped-to-remainder handoff and mixed column groups, an `attn.core` largest-leaf error, and
receipt wording. The second pass (`/tmp/opencode/review-group2.log`) accepted the handoff,
mixed-group and mutation coverage, the gate rows and the projection ceiling, and refused:

- the nine-column pooled-dispatch test never reached the wide tile on a multi-participant pool
  (fixed: `NQ_TILE_COLS × pool width` columns, plus an assertion that `nq_act`'s scales are the
  grouped quantizer's, re-gated in `group-gate4.log` and re-measured in `group-final2.log`);
- the mutation paragraph cited the pre-fix log (fixed and re-run in
  `/tmp/opencode/group-final2.log`);
- `Ffn.lean`, `Ffn2.lean` and `Gemm7.lean` do not unfold or audit `chart.quant_act*` (list
  corrected);
- the floor used an unstated 3.20 s (fixed: `6.208 − 3.138 = 3.07 s` from the final split);
- "same-window 1.62×" divided this window by llama's earlier row (reworded to 1.61× with the
  window factor named);
- the workspace test command stopped at the `--lib` failure (stated in Omissions).

The refusals that remain are the unrun maintainer gate, the unmeasured laptop, the uninterleaved
llama.cpp row, the 0.6B-only quality screen, and the `code_select_twin` extraction debt; each is in
Omissions.

Mutation controls on the box copy of the source, each targeted test run, then the source restored
and md5-checked (`/tmp/opencode/group-final2.log`, tail):

- **Group magnitude.** `let end = ((block + run) * ACT_QK).min(x.len());` → `(base + ACT_QK)` keeps
  only the first block's magnitude in the group scale. `quant_act_grouped_repeats_the_scale_and_keeps_the_contract`
  fails (`mutA-exit=101`, the second-block spike refuses); restored, it passes and `chart/mod.rs`
  returns to `87d1276a…`.
- **Run fold.** `pair[c] = _mm256_add_epi16(pair[c], _mm256_maddubs_epi16(select, plane));` →
  `pair[c] = _mm256_maddubs_epi16(select, plane);` keeps only the run's last block.
  `the_wide_source_folds_equal_block_runs` and `the_grouped_scale_reaches_the_pooled_reader` both
  fail (`mutB-wide-exit=101`, `mutB-pooled-exit=101`); restored, both pass and `kern/nq.rs` returns
  to `c2781526…`.

## Commands

```sh
# the final build, PPL pair, paired wall, turn, phase split, profile and mutations
# (the local scripts are /tmp/opencode/group-*.sh; the box runs them from /tmp/):
ssh ubuntu@139.99.223.161 '~/bin/adg run -- /tmp/group-final2.sh'
ssh ubuntu@139.99.223.161 '~/bin/adg run -- /tmp/group-gate4.sh'
ssh ubuntu@139.99.223.161 '~/bin/adg run -- /tmp/qk-sweep8k.sh'

# cold prefill A/B, three interleaved rounds at width 8 (inside group-final2.sh)
APEIRON_THREADS=8 APEIRON_PREFILL_COLS=1024 ~/bin/prefix_solve.final \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len 1827 --differ 205 --decode 0 --mb 440 | grep '^cold'

# the phase split, one run per arm; the split prints before its `cold` wall line
APEIRON_THREADS=8 APEIRON_PREFILL_COLS=1024 ~/bin/prefix_solve.final \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len 1827 --differ 205 --decode 0 --mb 440 --phases 2>&1 | grep -B20 '^cold'

# the profile (perf record around a copy of the same binary, then the symbol split)
perf record -q -F 499 -g -o /tmp/perf-final.data -- /tmp/prefix_solve.profile \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len 1827 --differ 205 --decode 0 --mb 440
perf report --stdio -i /tmp/perf-final.data --percent-limit 0.4 --sort symbol

# quality pair, 8,176 tokens, one window
APEIRON_THREADS=8 /tmp/ppl.head --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --data ~/models/wiki.test.raw --tokens 8192 --ctx 512 --head-batch 16 --kernel q8act
APEIRON_THREADS=8 ~/bin/ppl.final --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --data ~/models/wiki.test.raw --tokens 8192 --ctx 512 --head-batch 16 --kernel q8act

# the turn, three interleaved pairs (inside group-final2.sh)
APEIRON_THREADS=8 APEIRON_PREFILL_COLS=1024 ~/bin/prefix_solve.final \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len 1827 --differ 205 --decode 64 --mb 440 | grep '^cold'

# the kernel microbenchmark, checked against its scalar selection
rustc -O -C target-cpu=native /tmp/opencode/nqdot3.rs -o /tmp/nqdot3 && /tmp/nqdot3 3
```

Logs `/tmp/opencode/group-final2.log` (the final build, the PPL pair, the paired wall, the turn, the
phase split, the profile and the two mutations), `/tmp/opencode/group-gate4.log` (the contributor
gate), `/tmp/opencode/qk-sweep8k.log` (the design screen), `/tmp/opencode/review-group.log` and
`/tmp/opencode/review-group2.log` (the two reviews). Scripts `/tmp/opencode/group-final2.sh`,
`/tmp/opencode/group-gate4.sh`, `/tmp/opencode/qk-sweep8k.sh`, `/tmp/opencode/ppl-head-pair.sh`.
Source `/tmp/opencode/nqdot3.rs`.

## Omissions

- The laptop is not measured. Every wall is box `measured`; the box has 8 × EPYC Milan and 256 MiB
  L3, the laptop 4 × i5-10210U and 6 MiB, so no ratio here is a laptop claim. No `cargo` or `mdbook`
  gate ran on the laptop.
- llama.cpp was not interleaved in this window. Its 4.00 s row is the previous receipt's measured
  value, and the ratio here applies this window's factor from the old binary; it is `measured` in
  two windows, not one interleaved comparison.
- The maintainer gate is unrun and required. The extraction's treatment of `quant_act_grouped` is
  unverified.
- The `--all-features` test suite needs the `glslc` pin and a GPU; neither host has them.
- The box had just released another lane's guarded job, so the absolute walls are above the shipped
  6.957 s; the interleaved ratio 0.899 is the graded number, and the clean-equivalent 6.25 s is a
  scaled value, not a run.
- The phase splits and the profile are single runs, not paired; only their shares are read.
- `cargo test --workspace --locked` aborted at the `apeiron --lib` failure, so the workspace's
  integration, example and doc targets did not run in that invocation; the `--lib` run is the 679
  passed, 1 failed row above.
- The performance-gate instrument is `prefix_solve --phases`; `examples/phase_split --target`
  traces a decode token and is not run against this prefill change.
- The turn is a 1,827-position prefill plus 64 decoded tokens at 2k context, not a 1,000-token turn
  at 200k; its prefill mean is a separate set of runs from the `--decode 0` pair.
- Quality is the NQ path on the 0.6B v3 pack with the `q8act` reader over 8,176 tokens. The chart
  FFN and head and the `matmat`/`gemm` fixtures keep per-block scales and are not covered by the
  grouped pair.
- The 7B pack and 200k context are not on this box; T2 at 200k and M1 are untouched.
