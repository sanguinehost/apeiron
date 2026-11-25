# nq: one dispatch per linear

Criterion: **T2** (turn latency), partial: the cold 1,827-position prefill of the 0.6B pack and its
64-token turn, at block 1024. It is not a 1,000-token turn at 200k, and not a 7B result.
Grades: `measured` for every wall, phase split, fault and cycle row; `tested` for the pooled
equality tests and the mutation falsifier. Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU
EPYC Milan, 8 sockets × 1 core × 1 thread, 31.3 GiB, 256 MiB L3, 2.4 GHz base, no AVX-512, no VNNI),
no GPU. Times UTC. The laptop — the only device a timed claim may name — is **not measured**;
every wall below is box `measured` (Omissions).

Model: `~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf`; pack
`~/models/nq-scratch/nq06-v3.nqp` (196 linears, 28 layers, 0.977441 b/w, `k = 0` salient,
`res_layers = 0`); harness `examples/prefix_solve.rs`, `--len 1827 --differ 205 --decode 0 --mb
440`. Every timed run under `~/bin/adg run --` (16 GiB cap, 8 jobs); `touch src/lib.rs` before the
timed build. The rebuild and section comparison below run inside the same guard, with one ungraded
cold smoke row.

## The baseline

HEAD is `db5b42f9` and its `src/engine/kern/nq.rs` is md5 `bd5e02dd…`. Its binary is
`~/bin/prefix_solve.final5`, md5 `d62ad6d4…`, built in the session that committed the retained
planes (`bench/cpu/1795593118-nq-planes.md`). The first screen in this session used
`~/bin/prefix_solve.head` (`3886b48e…`, `nq.rs` `b9a1004f…`), which is commit `0f65d137`, one commit
**before** the planes change: its 10.4 s rows are not HEAD and none of them is used below. In the
window below `final5` is 7.854 s and the change 6.957 s at width 8; the difference between the two
"heads" is the committed planes change, not this one.

## What changed

`src/engine/kern/nq.rs`; `src/engine/loader/nq.rs` gets one doc-comment correction
(`nq_proj_pooled` now says one dispatch, not one `V` and one `U`), with no code change. The
readers, the dot bodies, the residual add and the per-column arithmetic are untouched.
`src/engine/step/cols.rs` is byte-identical to HEAD (md5 `dba527b5…`).

**A pooled linear is one parallel region.** Under the `Q8` reader the staged body crossed the pool
six times per linear: `t` build, activation quantization, `V` fold, the `rank·max|z|` bound scan,
`z` quantization and `U` fold, with each column's reader built on the dispatching thread between
stages. `ColPipeline` now runs a band's whole pipeline inside the task that owns the band — `t`
words, `quant_act`, `V` dot, the `z` bound and quantization, residual readers, `U` dot — so a linear
has one barrier and every per-column word is written by the participant that reads it. The band is
`⌈k/width⌉` columns, rounded up to a multiple of eight when it holds a group (`k = 1827`, width 8 →
229 → 232), so the wide tile still runs on every column of a full band and the last band carries the
remainder; below one group the band stays `⌈k/width⌉` columns, one when `k ≤ width`. The `U` stage
writes each column straight into its `Vec`; no flat output plane is allocated or copied.

The shape is llama.cpp's at `049326a00`, read from the source and the emitted code:
`ggml_compute_forward_mul_mat` (`ggml/src/ggml-cpu/ggml-cpu.c:1254`) gives each thread a row chunk,
the thread converts its `src1` rows to `vec_dot` type inside that chunk and calls `llama_sgemm`, and
the pass ends on one `ggml_barrier`; `tinyBLAS::gemm`
(`ggml/src/ggml-cpu/llamafile/sgemm.cpp:583`) starts every thread on its own tile and hands out the
rest from one job counter, with no barrier between tiles. Our six dispatches were the difference
the emitted code does not have: llama.cpp's whole GEMM is one region per pass.

## The residual add, measured and not shipped

A band-parallel `add_cols` was built and timed in the same window. It buys **0.335 s at eight
participants** (6.957 → 6.622 s, 0.952×), 0.23 s at four and 0.30 s at one. Decode's `k = 1` takes
the serial branch in both arms; the measured turn decode is 2.073 s shipped against 2.099 s with the
add. It is **reverted**, because `step::cols::add_cols` is extraction source:
`proof/ApeironWord.lean:55294` defines `step.cols.add_cols` and its loop, and
`proof/corpus/Serve/Block/Cols2.lean:534` proves `word_add_cols_refuses` — the loop stops at the
first column whose checked add refuses and leaves every later column untouched. A pool of bands runs
every band, so the `Err` post-state is not the proved one. Preserving it would need a bridge change,
not a Rust-only one; the measured 0.335 s is left on the table for that follow-up. The unshipped arm
is `~/bin/prefix_solve.merge3` (`aa1a6448…`, `nq.rs` `148717b5…` with `cols.rs` `653a594f…`), and its
rows are in the table below so the size of the deferred change is on the record.

## Result

### Cold prefill: paired A/B against HEAD

HEAD = `final5` (`d62ad6d4…`); shipped source = `src/engine/kern/nq.rs` md5 `56b92a8f…` and `cols.rs`
`dba527b5…`; the timed binary is `~/bin/prefix_solve.merge4`, md5 `eae8ce74…`, whose `.text` is
identical to the shipped source's rebuild `ship4` (Gates). Three interleaved rounds at
width 8, two at widths 4 and 1, `APEIRON_PREFILL_COLS=1024`, 1,827 positions, box load 0.23 at the
window's start with no other job running (`/tmp/opencode/final2-timed.log`). The third column is the
unshipped parallel add, timed in the same rounds.

| participants | HEAD s | shipped s | shipped/HEAD | + add s | + add/HEAD | spread (HEAD/shipped/add) |
|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 42.086 | 44.955 | **1.068** | 44.650 | 1.061 | 1.0% / 2.1% / 1.6% |
| 4 | 12.090 | **11.256** | **0.931** | 11.030 | 0.912 | 1.6% / 2.1% / 0.03% |
| 8 | 7.854 | **6.957** | **0.886** | 6.622 | 0.843 | 4.3% / 2.6% / 2.0% |

At eight participants the three shipped/HEAD pairs are 0.920, 0.864, 0.875 (mean 0.886), and the
shipped rates are 232.62 → **262.61 tok/s**. At four the two pairs are 0.914 and 0.948 (mean 0.931).
Width 1 is a **6.8% regression** (pairs 1.063 and 1.074), and it is the pipeline, not the add: the
add arm is 5.8% and 6.4% above HEAD, so the two change arms span 5.8–7.4%. One participant runs all
1,827 columns in one band, where the staged body resized each column's output on the dispatching
thread and built the readers in a pass of its own; which part of the 6.8% is the band's per-column
reader vectors is not separated. Widths 4 and 8 are the target.

### Phase split, width 8, both arms in one window

`--phases`, one cold run per arm, adjacent runs (`/tmp/opencode/final2-timed.log`). These are single
runs; the paired wall above is the graded number.

| phase | HEAD ms | share | shipped ms | share | shipped/HEAD |
|---|---:|---:|---:|---:|---:|
| `ffn` | 2770.64 | 35.7% | 2377.82 | 34.8% | 0.858 |
| `attn.core` | 1789.26 | 23.1% | 1755.92 | 25.7% | 0.981 |
| `attn.proj_in` | 1407.06 | 18.2% | 1069.37 | 15.7% | 0.760 |
| `attn.proj_out` | 528.99 | 6.8% | 466.35 | 6.8% | 0.882 |
| `norm` | 448.30 | 5.8% | 316.53 | 4.6% | 0.706 |
| `residual` | 396.03 | 5.1% | 431.02 | 6.3% | 1.088 |
| `pool.wait` | 240.12 | 3.1% | 225.03 | 3.3% | 0.937 |
| `pass` | 7372.50 | 95.1% | 6450.06 | 94.5% | 0.875 |
| cold wall | 7.751 s | | 6.827 s | | 0.881 |

The three projection leaves are **4,706.70 → 3,913.54 ms (0.831)**, 793 of the 924 ms the wall
falls (86%); `attn.core` is 0.981, which no part of the change targets. `norm` 0.706 and `residual`
1.088 move with no change to either, in opposite directions: a single-run split cannot apportion
that, so neither is attributed.

The change is prefill-only, so `examples/phase_split --target` is not its instrument: that tool
traces decode, and decode is flat after a short prefix and 6.6% worse after a packed one (below).
The prefill instrument is `prefix_solve --phases`; its largest leaf is `ffn` at 35.7%, and the target
group — `ffn`, `attn.proj_in`, `attn.proj_out`, the three callers of the pooled projection — is
60.7% of the HEAD wall.

The projection leaves were 60.7% of the HEAD wall and are 57.3% after, so the ceiling a further
projection change can reach is **`1/(1 − 0.573) = 2.34×`** (`2.55×` before). This change removes
16.9% of the projection leaf time (4,706.70 → 3,913.54 ms); the rest is the kernel's per-term cost.

### Turn and decode

`--len 1827 --differ 205 --decode 64`, width 8, three interleaved rounds in the same window
(`/tmp/opencode/final2-timed.log`): prefill 7.652 → **6.753 s** (0.882×, 238.75 → 270.55 tok/s). The
64 decoded tokens are **1.944 → 2.073 s (32.92 → 30.87 tok/s, 1.066×)**, and every round moves the
same way (1.039, 1.128, 1.033). The turn's total is **9.596 → 8.826 s (0.920×, −8.0%)**. The
prefill half is the win; the decode half is a **measured 6.6% post-prefill cost**, and the same
phase costs 2.5% in window 1 and 7.8% in the add-arm window — the sign is consistent across three
windows, so it is not a one-off.

The per-token work itself is not slower. After an **8-token** prefill, 64 tokens are 1.077 s HEAD
and 1.056 s change (0.981×, two pairs, `/tmp/opencode/final-attr.log`): with almost no prefill in
front of it, decode is flat. That control and the traced split below ran on the add arm, whose
decode takes the same serial residual loop as the shipped source's, so they bound the shipped
binary too. The traced decode split at a 1,827-token prefix is 29.05 ms/token HEAD and 29.81
ms/token change, with `attn.core` 12.455 → 12.336 ms, `ffn` 6.949 → 7.542 ms, `attn.proj_in` 3.311
→ 3.516 ms, `attn.proj_out` 2.228 → 2.349 ms and `pool.wait` 3.012 → 3.271 ms. **The cause is not
isolated**: the short-prefix control says the per-token path is unchanged, the turn says something
after a packed prefill costs 2–8%, and the candidates are allocator state and code layout left by
the prefill. It is not attributed to a line, and it is not presented as neutral. Decode's dominant
term is `attn.core` at 41.4–42.9% (ceiling 1.71–1.75×), not the projection path.

### Cycles, instructions, branches, width 8, 1,827 positions

`perf stat -e cycles,instructions,branches`, one cold run per arm in the same window
(`/tmp/opencode/final2-timed.log`):

| arm | cycles | instructions | IPC | branches | cold prefill |
|---|---:|---:|---:|---:|---:|
| HEAD | 263.27 G | 676.03 G | 2.57 | 73.10 G | 7.921 s |
| shipped | **232.15 G** | **630.40 G** | **2.72** | **69.55 G** | 7.094 s |

Cycles fall **−11.8%**, instructions **−6.7%**, branches **−4.9%**, and IPC rises 5.8%: the pass
does less per-column work and spends less time at barriers, so the same eight sockets retire more
per cycle. The per-function split on the change's `nq.rs` with the unshipped add in `cols.rs`
(`perf record`, 1,827 positions, width 8,
`perf report --sort symbol`, which sums worker and caller threads;
`/tmp/opencode/ship3-gate.log`): `code_select_row8_avx2` 27.0%, the pool's parked-worker frame
17.1%, `attn_core_tile_avx2_inner` 8.6%, `gemm_tile` 8.3%, `project_row` 7.9%, `quant_act` 5.5%,
`ColPipeline::run` 5.2%, `swiglu_f32_avx2_8_into` 2.4%, `rms_norm_fixed` 2.2%, `StageCols::dot_row`
1.8%. llama.cpp in the same window: `gemm_bloc<4,3>` 50.4%, flash-attention 26.6%, fp16→fp32
conversion 7.4%, libgomp 5.1%, fp32→fp16 1.8%, softmax 1.7%, RMSNorm 1.6%, RoPE 1.0%. The two
projections differ in data type (F16 FMA against int8 sign selection), so the same-function
comparison is the emitted instruction counts below, not the cycle shares.

### Faults and resident set, width 8, 1,827 positions

`/usr/bin/time -v`, one run per arm in the same window (`/tmp/opencode/final2-timed.log`):

| arm | minor faults | major | peak RSS KiB |
|---|---:|---:|---:|
| HEAD | 296,415 | 0 | 1,363,812 |
| shipped | 297,797 | 0 | 1,372,652 |

Peak RSS moves **+8,840 KiB (+0.65%)**, from 1.301 GiB to **1.309 GiB**: neutral at this length. The
+83.9 MiB in the first screen was the committed planes change, already in HEAD
(`bench/cpu/1795593118-nq-planes.md`), not this one. M1 is unchanged by this receipt.

### Position against llama.cpp

Same box, pin `049326a00`, F16 `Qwen3-0.6B-Base.f16.gguf`, `-p 1827 -n 0 -t <w> -r 1`, interleaved
between the engine rounds in the same window (`/tmp/opencode/final2-timed.log`):

| participants | llama.cpp tok/s | engine shipped tok/s | ratio | HEAD ratio |
|:---:|---:|---:|---:|---:|
| 1 | 73.99 (73.59–74.40) | 40.64 | 1.82× behind | 1.70× |
| 4 | 258.83 (257.87–259.79) | 162.32 | 1.59× behind | 1.71× |
| 8 | 456.83 (449.56–464.31) | 262.61 | **1.74× behind** | 1.96× |

At width 8 the llama wall is **4.00 s** (3.94–4.06) and ours is **6.957 s**. **The target was 1× or
0.5×; neither is reached.** Parity needs another **1.74×**; half needs **3.48×**. This change moves
the same-window ratio 1.96× → 1.74×. The remaining 1.74× is a kernel question, not a dispatch one:
parity sits inside the 2.34× the projection share allows, but with the non-projection floor left at
**2.91 s** the projection leaves would have to fall from 3.91 s to about 1.09 s, **3.6×** their
current rate, and llama's entire prefill is 4.00 s. Half of llama (2.00 s) is **below that 2.91 s
floor**, so 0.5× on this pack and host is refused by the share ceiling unless `attn.core` (1.76 s)
gives up about 0.9 s of the floor as well.

## The emitted code

The pinned llama.cpp commit's F16 GEMM for this host is `tinyBLAS<8, __m256, __m256, float, float,
float>::gemm<4,3,4>`: 924 instructions after the inline, 180 `vfmadd231ps`, 177 `movq`, 122
`vmovups`, 66 `leaq` (`bench/cpu/1795601655-llama-f16-4x3x4.s`, from
`/tmp/opencode/emit-llvm.sh`). Our eight-column sign kernel `code_select_row8_avx2` is 836
instructions, 168 `mov`, 59 `cmp`, 53 `vpaddq`, 35 `vextracti128`, 34 `vpmovsxdq`
(`bench/cpu/1795601655-nq-row8.s`, from `/tmp/opencode/emit-kernels.sh`; the file has 850 lines, of
which 843 are instruction lines: 836 and seven trailing `int3`). That listing is the shipped
binary's sequence: disassembling `code_select_row8_avx2` from `~/bin/prefix_solve.merge4` gives the
same 836 instructions in the same order (`/tmp/opencode/row8.merge4.s`; the final source's `ship4` build has the same `.text`), with only rip-relative
displacements and branch targets differing because the link addresses differ. The loop difference
the port removes is not in either kernel: it is around them. llama.cpp enters its kernel once per row
chunk from one parallel region; under the `Q8` reader the staged body crossed the pool boundary six
times per linear, building the per-column readers on the dispatching thread between quantization
and fold.

## The claiming counter, ported and refuted

`tinyBLAS::gemm` does not give each thread a fixed tile: every thread starts on its own index and
takes the next with `ggml_threadpool_chunk_add(1)` (`sgemm.cpp:596-627`). Ported into
`chart::pool::for_tiles3` with the same band arithmetic, three interleaved pairs at 1,827, width 8
(`/tmp/opencode/tiles-gate.log`): static 6.682 / 6.634 / 6.699 s (mean 6.672), claiming 6.743 /
6.924 / 6.900 s (mean 6.856) — **1.028× worse**, non-overlapping. At width 1 the claiming counter
wins (41.828 s against 44.221), and at 512 positions it won in the earlier screen (1.720 against
1.772 s). The headline size and the target width lose, so the helper and its export are removed;
the comment in `project_pooled_body` records the refutation. The cause is not separated: this pool
publishes the epoch and the caller serves its own partition immediately, so the caller can take
tiles a parked worker would have served, where llama.cpp touches the counter only with every
participant awake.

## Gates and falsifiers

Contributor gate on the box copy of the shipped source (`md5 nq.rs 56b92a8f…`, `loader/nq.rs`
`4bc4f936…`, `cols.rs dba527b5…`, book `46f9431d…`), no GPU and no `glslc` pin
(`/tmp/opencode/ship11-gate.log`); `fmt`, `clippy`, the Rust tests and `mdbook` ran on this source
and this book text:

The timed binary `merge4` (`eae8ce74…`) was built from `nq.rs` `148717b5…`. A later
`objcopy --dump-section` call without an output file overwrote it in place; it was restored
bit-for-bit from `merge5`, the rebuild `/tmp/opencode/ship4-gate.log` records as identical
(`binary-identical=yes`, same md5). The shipped source `56b92a8f…` differs from the timed source in
comments only: the module-level `Parallelism` block, which describes `ColPipeline` and is one line
shorter than the timed source's (so every later `nq.rs` line is one less), the
`ggml_compute_forward_mul_mat` pointer (defined in `ggml/src/ggml-cpu/ggml-cpu.c:1254`, not in
`ops.cpp`, which dispatches it), the one-dispatch counts and band wording in `project_pooled` and
`project_pooled_body`, and `loader/nq.rs`, whose `nq_proj_pooled` doc no longer says one `V` and one
`U` dispatch. Rebuilt as `~/bin/prefix_solve.ship4` (`000d9e8a…`) and compared
section by section (`/tmp/opencode/ship11-gate.log`): every SHF_ALLOC section is byte-identical
except `.note.gnu.build-id` and 90 bytes of `.data.rel.ro` — the low byte of
`core::panic::Location::line` in its 24-byte `{ptr, len, line, col}` records for
`src/engine/kern/nq.rs`, each exactly one less because the module block lost a line; the record at
`0x203218` reads line 2776 in the timed binary and 2775 in `ship4`, column 25, file
`src/engine/kern/nq.rs` (`/tmp/opencode/ship11-datarel.log`, run directly on `ship4`). `.text`
(`91e830af…`), `.rodata` (`55d63d7c…`), `.rela.dyn`, `.eh_frame`, `.data` and `.init_array` are
identical, and `code_select_row8_avx2` disassembles to the same 845 extracted lines in both (843
instruction lines: 836 instructions and 7 trailing `int3`); its instruction text equals the
committed listing's after comment removal and hex normalization
(`/tmp/opencode/ship11-gate.log`, where the box copy carries the listing as `/tmp/row8.committed.s`).
The committed file is 850 lines: six lines of `objdump` preamble and the 844-line symbol slice, its
header plus 843 instruction lines. Every timed row of the shipped
arm ran this machine code; the add arm is the separate unshipped binary above. A cold smoke run of
`ship4` after the comparison (not a graded row) prints 7.111 s and first token 315.

| gate | result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | exit 0 |
| `cargo test --lib --locked kern::nq` | 22 passed, 0 failed, 4 ignored |
| `cargo test --lib --locked engine::step` | 19 passed, 0 failed, 3 ignored |
| `cargo test --workspace --locked` | 676 passed, 1 failed: `gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl`, which needs the `glslc` pin this box does not have; no other failure (`/tmp/opencode/ship11-gate.log`). This is the default-feature workspace run, not `--all-features`; the GPU suite needs the pin and a device and runs on neither host |
| `mdbook build book`, `mdbook test book -L $HOME/apeiron-mdbook/debug/deps` | pass, on the final book md5 `46f9431d…` (`/tmp/opencode/ship11-gate.log`); an earlier pass on `533ed8be…` is `/tmp/opencode/book-final.log` |
| `python3 tools/check-claims.py`, `python3 tools/book-check.py` | pass (laptop) |

The maintainer gate was **not run**: neither host has the pinned Aeneas, Charon and Lean toolchain.
It is **required** for this change, not exempt. `proof/regen.sh` extracts the call graph of
`word_engine_prefill_state`, and that closure reaches `src/engine/kern/nq.rs`: `step/cols.rs:507`
(`nq_proj_pooled`) calls `loader/nq.rs:224` (`EngineLoader::nq_proj_pooled`), which calls
`project_pooled`/`project_q8_pooled`/`project_ternary_pooled` and then
`NqLinear::project_pooled_body`, while `proof/extract/lib.rs` pulls `src/engine/kern/mod.rs`, which
declares `pub(crate) mod nq;`; `regen.sh` already carries six `--opaque 'crate::kern::nq::*'`
entries. The committed `proof/ApeironWord.lean` predates that wiring — `git log -1 --
proof/ApeironWord.lean` is `7292279f`, regenerated at source `848a4ea7`, and `nq_proj_pooled`
arrives at `6b6e990f`, 210 commits later — and `proof/README.md` records the gate red at `6c724104`
with 415 hunks, so `proof/regen.sh --check` would report pre-existing drift on HEAD as well and
cannot isolate this change. `ApeironWord.lean` freshness and the Charon/Aeneas translatability of
`ColPipeline` are **unverified** here: the change adds `Task3`/`for_chunks3`, an `AtomicBool` not
listed in `proof/boundary-axioms.txt`, `std::array::from_fn`, local closures and `.map().collect()`
inside the `word_engine_prefill_state` closure, which is what the gate would test. The route is
`proof/bootstrap.sh`, `proof/regen.sh --check`, then
`cd proof/corpus && lake build`, whose Mathlib build is several GB and did not run on either host.
The `cols.rs` revert removes the change that would have contradicted a proved post-state
(`word_add_cols_refuses`), but it does not close the gate. The gate is an open blocker on this
change, not a waived check.

Review: `gemini-4-hydrogen-high` (Hydrogen) reviewed the change across
`/tmp/opencode/review2.log`–`/tmp/opencode/review7.log`. The refusals it
leaves open are the three above: the unrun maintainer gate, the unmeasured laptop, and the
`--all-features` suite that needs the `glslc` pin. Its line-level findings — stale dispatch counts in
`nq.rs`, the band-width sentence, the committed-listing line count, the gate-log and datarel
citations, and the missing `cd ~/llama.cpp` in the commands — are fixed and re-gated by `ship11`.

Mutation falsifier on the shipped source (`/tmp/opencode/ship11-gate.log`: `nq.rs` `56b92a8f…`,
`cols.rs` `dba527b5…` before and after). Applied in the box copy, restored and
`md5`-checked:

- **Band origin.** `let c0 = task * self.cols;` → `let c0 = 0;` makes every band recompute the first
  band's columns. `the_pooled_block_is_bit_equal_at_every_block` fails (`nq.rs:3518`,
  `mutated-exit=101`); restored, it passes (`restored-exit=0`) and both source md5s are the shipped
  ones. The same mutation was run earlier against `nq.rs` with the unshipped add in `cols.rs` and
  failed the same way (`/tmp/opencode/mutate2.log`).

## Commands

```sh
# the paired A/B, phases, turn, RSS, cycles and llama rows, on the box copy of 148717b5 / dba527b5:
ssh ubuntu@139.99.223.161 '~/bin/adg run -- bash /tmp/final2-timed.sh'
ssh ubuntu@139.99.223.161 '~/bin/adg run -- bash /tmp/ship11-gate.sh'

md5sum ~/bin/prefix_solve.final5 ~/bin/prefix_solve.merge4 ~/bin/prefix_solve.merge5 ~/bin/prefix_solve.ship ~/bin/prefix_solve.ship2 ~/bin/prefix_solve.ship4 ~/bin/prefix_solve.merge3
md5sum src/engine/kern/nq.rs src/engine/loader/nq.rs src/engine/step/cols.rs

# cold prefill A/B, three interleaved rounds at width 8 (inside final2-timed.sh)
APEIRON_THREADS=8 APEIRON_PREFILL_COLS=1024 ~/bin/prefix_solve.merge4 \
  --model ~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf \
  --pack ~/models/nq-scratch/nq06-v3.nqp \
  --tokenizer ~/models/qwen3-0.6b-base-e8/tokenizer.json \
  --len 1827 --differ 205 --decode 0 --mb 440 | grep '^cold'

# llama.cpp row, same box and window
~/llama.cpp/build/bin/llama-bench -m ~/models/qwen3-0.6b-base-f16/Qwen3-0.6B-Base.f16.gguf \
  -p 1827 -n 0 -t 8 -r 1

# emitted kernels, with the comparison's flags (from ~/llama.cpp)
cd ~/llama.cpp
FLAGS="-DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED \
  -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 \
  -Iggml/src/.. -Iggml/src/. -Iggml/src/ggml-cpu -Iggml/src/../include \
  -O3 -DNDEBUG -std=gnu++17 -march=native -fopenmp -Wno-everything"
clang++ $FLAGS -S -emit-llvm -o /tmp/llvm/sgemm.ll ggml/src/ggml-cpu/llamafile/sgemm.cpp
objdump -d --no-show-raw-insn --disassemble=code_select_row8_avx2 target/release/examples/prefix_solve

# the 90 differing .data.rel.ro bytes, decoded against the nq.rs Location string
python3 /tmp/opencode/ship8-datarel.py /tmp/opencode/prefix_solve.merge4 /tmp/opencode/prefix_solve.ship4
```

Logs `/tmp/opencode/final2-timed.log` (the three-arm paired A/B, turn, phases, RSS, cycles and the
interleaved llama rows), `/tmp/opencode/ship11-gate.log` (the gate, mdbook, all-SHF_ALLOC section
comparison, `row8` disassembly, committed-listing comparison, falsifier and smoke on the final
source), `/tmp/opencode/ship11-datarel.log` (the 90-byte `.data.rel.ro` decode on `ship4`),
`/tmp/opencode/ship10-gate.log` (the earlier gate on `nq.rs` `6bd5a7cd…`), `/tmp/opencode/ship9-gate.log`
(the earlier gate on `nq.rs` `29ac6115…`), `/tmp/opencode/ship8-compare.log`
(the earlier all-section comparison of `ship`), `/tmp/opencode/ship4-gate.log` (the earlier gate on
`148717b5…`), `/tmp/opencode/ship3-gate.log`
(the earlier gate, and the `perf record` splits — same `nq.rs`, the unshipped add in `cols.rs`),
`/tmp/opencode/ship3-timed.log` (the add arm's turn and phase rows), `/tmp/opencode/ship5-mdbook-mutate.log`
(the earlier `mdbook` pass and the falsifier), `/tmp/opencode/mutate2.log` (the
same falsifier with the unshipped add in `cols.rs`), `/tmp/opencode/final-attr.log` (window 1: the
short-prefix decode control and the decode split), `/tmp/opencode/book-final.log` (mdbook on the
earlier book text),
`/tmp/opencode/tiles-gate.log` (the counter refutation). Scripts `/tmp/opencode/final2-timed.sh`,
`/tmp/opencode/ship11-gate.sh`, `/tmp/opencode/ship8-compare.sh`, `/tmp/opencode/ship8-datarel.py`,
`/tmp/opencode/ship7-gate.sh`, `/tmp/opencode/ship5-mdbook-mutate.sh`, `/tmp/opencode/mutate.sh`,
`/tmp/opencode/final-attr.sh`, `/tmp/opencode/tiles-gate.sh`. The unshipped add's source is `/tmp/opencode/cols.parallel.rs`; the
instruction listings are `bench/cpu/1795601655-nq-row8.s` and
`bench/cpu/1795601655-llama-f16-4x3x4.s`.

## Omissions

- The laptop is not measured. Every wall is box `measured`; the box has 8 × EPYC Milan and 256 MiB
  L3, the laptop 4 × i5-10210U and 6 MiB, so no ratio here is a laptop claim. No `cargo` or `mdbook`
  gate ran on the laptop (only `check-claims.py` and `book-check.py`); the Rust gates are the box's.
- The turn's 64 decode tokens are not a 200k-context turn; the 7B pack is not on this box. The
  prefill target is the 0.6B pack at 1,827 positions, not the 7B at 200k.
- Width 1 regresses 6.8% and is not fixed; the cause is the one band, not separated further. Widths
  2–3 are unmeasured.
- The split inside a band (weight read, bit expansion, dot, `quant_act`, reader build) is not timed
  separately; `perf record` gives the leaf shares above.
- The two phase splits are single runs, not paired; the paired wall is the graded number and the
  splits are read as shares.
- `norm`, `residual` and `attn.core` move with no change to those phases; a single-run split cannot
  apportion that, so no part of the saving is attributed to them.
- The box is quiet at the window's start (load 0.23 with no other job) but its own eight threads are
  in the load once a run starts. The three width-8 llama reps span 3.2% (449.56–464.31 tok/s), so
  the cross-engine ratio is quoted as 1.74× inside 1.71–1.77×. Absolute walls are `measured` in one
  window, not a multi-day mean.
- The timed binary's ELF md5 (`eae8ce74…`) is not the shipped source's rebuild (`000d9e8a…`). `.text`
  and every loaded section the arithmetic reads are byte-identical; the two differences are the
  build-id note and 90 panic-location line bytes, so no instruction in a timed row changed. The
  section log and the location decode are above.
- The residual add's 0.335 s at eight is measured but **not shipped**; its proof-boundary change is
  open, so the shipped change is the smaller one.
- Quality is not re-measured: the change is dispatch structure and column placement over the same
  readers and accumulation order. The evidence is the bit-equality tests above plus the first token,
  315 in every 1,827-token run across all three arms.
