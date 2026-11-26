# nq: the 0.6B cold prefill gap — 0.421 s outside the pass, 0.341 s inside, 0.304 s of caller wait

Criterion: **T2** (turn latency), partial: the cold 1,827-position CPU prefill of the 0.6B pack at
the engine's 913-column block (2 × 913 + 1), width 8, on the OVH box. The scope is
`1795664197-nq-prefill-gap.md` and `1795660694-nq-select-sign-dot.md`. This receipt changes no engine
code. It replaces the earlier draft's withdrawn 0.67 s decomposition. The measured facts are
**0.421 s of cold-turn work outside `Phase::Pass`** (one KV allocation, one cache clone, the extra
greedy head and the final dense head; only the allocation, clone, probe/drops, argmax/checksum and
the 5.886 ms extra greedy head lack a `pp1827` counterpart) and a **0.341 s pass gap** against llama's
median wall, plus
**0.304 s of caller barrier idle (`pool.wait`, turn-wide: 1,290 in pass + 5 in the two heads)**.
Subtracting that idle — a `model`, assuming workers keep up without moving work onto the caller —
leaves 4.013 s against llama's *total* wall of 3.977 s, **≥ +0.92%**. That is consistent with the work
side being near parity, not with an 8% deficit; it is not proof, because llama's own barrier wait is
unmeasured and the traced run is 47 ms above the pair median.

Grades: `measured` for every wall, phase, per-thread count and clock below; `model` for the
allocation/clone split (withdrawn, see below), the pass-minus-wait subtraction, the NQ band bound, the
per-symbol shares applied to totals (including llama's `cy.llama.data` table) and the ceilings, each
with its formula. No `tested` and no `proved`: no code changed, no test or Lean run. The raw lines are
in `1795674629-nq-prefill-serial.log`.

Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan, 8 sockets × 1 core × 1 thread,
31.3 GiB, 256 MiB L3, no AVX-512, no VNNI), no GPU. The laptop — the only device a timed claim may
name — is **not measured**. Times UTC; every job under `~/bin/adg run --` (16 GiB cap, 8 jobs, swap
zero). The box is shared; the pair is the graded wall, the trace run is the split.

**Base.** Repo HEAD `f736877a16bf5690006a1f35568e698d34f45cd6`; binary
`/home/ubuntu/bin/prefix_solve.cleanfin`, sha256 `57476450f79a8aae1cb2106749a90552ef754f90daaae4835d3f0e78690ea3fa`,
the treatment of `1795667807-nq-swiglu-q32-branch.md` built from a private clean copy of HEAD
`0b294ba7`. Pack `nq06-v3.nqp` `2e4670a4…`, skeleton `107c8edd…`, llama model `a33dce69…`. llama.cpp
`049326a00`. The per-thread, profile and trace windows differ; each table names its own.

## The result

The quiet window (06:45:23, load 1.31) ran llama and ours three times, interleaved. `spread` is
`(max − min) / min`; the mean is over the three runs:

| | run 1 | run 2 | run 3 | mean | median | spread |
|---|---:|---:|---:|---:|---:|---:|
| llama `pp1827`, t/s | 462.30 | 458.86 | 459.43 | 460.20 | 459.43 | 0.75% |
| llama, s | 3.952 | 3.982 | 3.977 | 3.970 | **3.977** | 0.75% |
| ours cold, s | 4.639 | 4.706 | 4.691 | 4.679 | **4.691** | 1.44% |
| gap, s | 0.687 | 0.724 | 0.714 | 0.709 | **0.714** | 5.44% |
| ratio | 1.174 | 1.182 | 1.180 | 1.178 | **1.180** | 0.69% |

The `--phases` run immediately after the pairs printed **4.738 s** cold (1.0% above the pair median,
+47 ms) and the cold-turn split below. The trace run is the one with the phase data, so the
decomposition is quoted on 4.738 s; the pair median is 0.714 s. The 47 ms drift is larger than the
36 ms residual that the wait subtraction leaves, which is why that subtraction is a consistency check
and not a closure.

| term | ms | share |
|---|---:|---:|
| cold turn | 4,738.000 | |
| `pass` | 4,317.241 | 91.1% of turn |
| outside `pass` | **420.759** | 8.9% of turn, **55.3% of the traced gap** |
| pass gap vs llama 3,976.667 | **340.574** | |
| caller `pool.wait`, turn-wide (1,290 in pass + 5 in two heads) | **304.154** | **≤7.05% of pass (6.42% of turn), ≤89.3% of the pass gap** |
| pass − wait | **4,013.087** | **≥ +36.420 ms vs llama, ≥ +0.92%** |
| in-pass untraced | 45.396 | 1.05% of pass |
| leaves inside pass (ex head) | 4,271.845 | 98.95% of pass |

The pass gap and the caller's wait agree to **36 ms (11%)**. That agreement is a consistency check,
not a decomposition. Read it as a bound: `pass − wait = 4.013 s` compares our wait-free pass with
llama's *total* wall, which includes llama's unmeasured barrier wait, and the subtraction charges the
5 head dispatches' wait to the pass, so 4.013 s is a lower bound on the zero-wait pass. The
over-subtraction is bounded by the two heads' combined wall: the traced greedy head is 5.886 ms and
the dense head is unmeasured. The residual against 3,976.667 ms is therefore **at least +36.420 ms
(+0.92%, `model`)**. The directly measured terms are 0.421 s outside `Phase::Pass` (total only) and
0.304 s of caller barrier idle turn-wide. Within the 0.421 s, the allocation, cache clone,
probes/drops, argmax/checksum and the 5.886 ms extra greedy head have no `pp1827` counterpart (the
final dense head does correspond to llama's one head); the claim that the prefill's *work* is
consistent with near-parity rests on the `model`, not on a paired work-to-work measurement.

## The pass and caller wait (0.304 s turn-wide)

`pool.wait` is the span around the caller's wait loop in `Pool::run_dyn`
(`src/engine/chart/pool.rs:592`): the caller runs its own partition (`run_partition(0, n, weighted)`)
and then spins on `remaining` until every published worker has finished. It is the caller's idle, not
the workers'. It sums to **304.154 ms over 1,295 dispatches**, 235 µs per dispatch. The count is
turn-wide: **1,290 dispatches inside `Phase::Pass`** — 2 × (28 × 14 + 1) = 786 at `k = 913` and
28 × 18 = 504 at `k = 1` — plus **5 outside it** in the two heads (3 in `head_greedy_state`, 2 in
`lm_head_fixed_sched`). The 5 head waits are bounded by the two heads' total wall — the traced greedy
head is 5.886 ms, the dense head is unmeasured — and the whole 304.154 ms is charged to the pass in
the subtraction in the table above and in the bound below, which over-subtracts by at most that
combined wall. The in-pass dispatch period is `4,317.241 / 1,290 = 3.35 ms`; the turn-wide 235 µs per
dispatch (**≤236 µs** if all 304.154 ms sat in the pass) is average idle within a dispatch, not a gap
between dispatches.

The leaves are the other side:

| leaf | ms | % of pass | calls |
|---|---:|---:|---:|
| `ffn` | 1,605.962 | 37.2% | 84 |
| `attn.core` | 1,508.851 | 34.9% | 84 |
| `attn.proj_in` | 611.349 | 14.2% | 84 |
| `attn.proj_out` | 243.869 | 5.6% | 84 |
| `norm` | 156.770 | 3.6% | 171 |
| `residual` | 135.071 | 3.1% | 168 |
| `embed` | 9.973 | 0.2% | 3 |
| untraced inside pass | 45.396 | 1.05% | |

The traced head (5.886 ms, 0.14% of the pass) sits outside the pass span (`head_greedy_state` is
called after `compose_cols`); it is part of the 420.759 ms outside, not a pass leaf.

Projection-bearing leaves (`ffn` + `proj_in` + `proj_out`) are **2,461.180 ms, 57.0% of the pass**;
`attn.core` is 34.9%; norm/residual/embed are 7.0%. The `pool.wait` overlay is nested inside these
leaves for **1,290 of its 1,295 calls** (5 sit in the two heads outside the pass), so the leaf
percentages include that idle; 4.013 s is a lower bound on the zero-wait pass.

The wait is a **straggler** wait: the caller's own partition is one of the eight tasks, and it waits
for the slowest of the other seven. Averaged over all 1,295 dispatches, 304.154 ms is 235 µs each,
which does not distinguish one bad dispatch from many small skews. The per-dispatch distribution and
the enclosing phase are not measured — the trace stores one total per phase, and `pool.wait` carries
no callsite. That histogram is the next instrument.

## The NQ band: 7 × 120 + 73

One systematic straggler is exact. `src/engine/kern/nq.rs:2626-2631`, `k = 913`, `width = 8`:

```text
band = ceil(913 / 8) = 115
cols = ceil(115 / 8) * 8 = 120          (NQ_TILE_COLS = 8)
for_chunks3 over 913 with chunk 120  ->  7 × 120 + 1 × 73
```

The eight tasks are not equal: seven carry 120 columns and the last 73. The fair share is
**114.125**; the seven heavy tasks carry **5.875 extra columns, 4.90% of a band**. At the projection
leaves' 2,461.180 ms that is **≤120.5 ms**. The bound is loose: the leaves also contain `SwiGLU`, the
`k = 1` pass, and fixed dispatch overhead/wait. A perfectly balanced integer split still leaves one
band at 115 columns (`7 × 114 + 115 = 913`), so the saving from balancing is
`(120 − 115) / 120 = 4.17%` of a band, **≤102.55 ms**; the 0.875-column excess of 115 over 114.125 is
unavoidable. The last participant's own spin is **39.16% of its cycles** in `pool::worker`, the same
as `(120 − 73) / 120 = 39.17%` unused band. That is consistent with the band being a major excess idle
source — **+16.55 own points over the six heavy workers' 22.61% baseline** — not an exact
decomposition: the spin is summed over every dispatch, not only the NQ projection, and the six-worker
baseline (22.61% of their own cycles, **2.82% of total each — 16.93% across all six**) is most of the
tail's 39.16% own-cycle spin. This is a work term (the band makes the seven heavy tasks longer),
separate from the 0.304 s wait; the tail finishing early is a symptom, not the straggler.

This is `nq.rs`'s band, not `pool.rs`'s split: the dispatch already publishes `n = 8` tasks, so
`live = min(7, n − 1)` and `range`/`round_robin_range` are not the cause. `range` is the weighted
contiguous split (`pool.rs:287`); the NQ dispatch uses the fixed 120-column chunk.

## The per-thread counters: what they do and do not price

`perf stat -p --per-thread` (attach at the `len 1827` banner) on `t1`, the post-banner process
lifetime. The printed walls are cold 4.862 + warm 0.127 + shared 0.143 = 5.132 s, against a caller
task-clock of 5.015 s, so **≥ 0.117 s** of caller wall is off-CPU (cause not measured here) and is
not in the cycle table; this is **not** the cold pass:

| participant | cycles G | instr G | IPC | task s | GHz | off-CPU vs caller s |
|---|---:|---:|---:|---:|---:|---:|
| caller | 18.302 | 42.351 | 2.31 | 5.0146 | 3.650 | 0 |
| six workers, mean | 16.291 | 39.909 | 2.45 | 4.4603 | 3.652 | 0.554 |
| tail worker | 15.793 | 30.972 | 1.96 | 4.3253 | 3.651 | 0.689 |
| total | 131.840 | 312.776 | 2.37 | | | |

`cycles ÷ task-clock` is 3.648–3.655 GHz on all eight. Therefore `cycles = clock × on-CPU seconds`,
and the caller's +2.011 G over the six-worker mean is **0.551 s of the workers' off-CPU time**
(5.0146 − 4.4603 = 0.554 s), not serial work: a spinning worker accrues cycles at the caller's rate,
so caller-serial wall cancels in the difference. The tail's +2.509 G is 0.687 s against 0.689 s
off-CPU. The earlier draft's "+2.00 G → 0.50 s serial glue" is withdrawn; the conversion was
arithmetic-wrong (`2.00 / 3.65 = 0.548`, not 0.50) and category-wrong (it is off-CPU, by identity).

The branch table is a separate run (`s1`, cold 4.869 s), not mixed into the cycle table: caller
54.413 M misses at 1.20%, six workers 28.5–29.1 M at 0.67–0.68%, tail 21.413 M at 0.65%. Total
248.266 M / 33.374 G branches = 0.744%. The `f736877a` per-symbol miss profile is in
`1795667807-nq-swiglu-q32-branch.md`; no per-thread miss profile is claimed here.

The idle loop is `r1` (a third window, 127.549 G cycles): **21.67% of all cycles are
`pool::worker`, and 99.5% of that symbol's samples are one instruction**, `1807ce: test $0x3ff,%ebx`,
the `spins.is_multiple_of(1024)` check in front of `spin_loop()`. The six workers' own share is
22.61%, the tail's 39.16%. `SPIN_BEFORE_PARK` is 5 ms (`pool.rs:86`). In the pass the dispatch period
is 3.35 ms (`4,317.241 / 1,290`), shorter than the park threshold, so the workers mostly spin there;
the `t1` post-banner-lifetime off-CPU of **0.544–0.689 s** (vs caller on-CPU; 0.66–0.81 s against the
5.132 s sum of printed turn walls; `t1.csv`'s 0.831 CPUs utilized puts perf's attach-to-exit wall at
6.03 s) is consistent with parking over the longer spans outside the pass (allocation, heads, cache
insert) and the warm/shared turns, and is not isolated to the cold pass. PMU skid on `pause`/`rep-nop` can move a sample to the following `test`, so 99.5% locates the
loop, not a cycle count of the `test` itself.

## The projection symbols: a `model` comparison

On `r1`'s own 127.549 G (not the 131.840 G of `t1`): `code_select_row8_avx2` 24.57% = 31.34 G and
`ColPipeline::run` 10.00% = 12.75 G, **44.09 G** together. llama's `gemm_bloc<4,3>` 50.63% +
`ggml_fp16_to_fp32_row` 7.57% of 116.6 G = **67.86 G**. Both totals are `model`: different windows,
and llama's 116.6 G is `1795664197`'s two-point slope, not a paired capture. The difference,
**−23.8 G**, is consistent with the pass work being **≥ +0.92% above llama's total wall** (`model`),
but it is not a paired measurement.

The bucket boundaries are stated: `ColPipeline::run` includes the checked `i128` epilogue, so the
epilogue is in projection, not glue; `quant_act_grouped_avx2` (1.85%) is the activation quantizer and
is counted separately; llama's bucket includes its F16→F32 weight conversion. The claim is narrow and
`model`: on these different windows and totals, the projection *symbols* are cheaper, and the pass
*work* side is consistent with near-parity — **≥ +0.92% against llama's total wall**, with llama's
wait and the five head waits both unmeasured (the head waits bounded by the two heads' walls); the
0.304 s of caller wait is barrier idle rather than projection arithmetic, and which enclosing phase
owns it is unmeasured. It is **not** an exoneration of the dot: 57.0% of the pass wall and up to
120.5 ms of NQ band skew sit inside the projection leaves, and no paired `cycles:u` capture of both
binaries exists. That same 57.0% and the `1795664197` floor (all glue at zero: projection + attention
**252.6 G instructions** = 3.93 s at the measured rates, `model`; 0.98× `1795664197`'s 4.004 s llama
wall, 0.99× this window's 3.977 s) bound any win past parity.

## Outside the pass: 0.421 s measured, contents unsplit

The timer in `examples/prefix_solve.rs:97-103` covers `longest_prefix` + `prefill_with_cache_mode`.
The pass spans only `compose_cols`; the interval outside it contains, in order
(`src/engine/prefix/mod.rs:538-618`):

1. the prefix probe (`examples/prefix_solve.rs:98`, then `prefix/mod.rs:558`), both before the
   allocation;
2. `decode_state(1835)` — the 421 MB KV state. `KvState::with_carrier` allocates one zeroed plane and
   `KvPlane::clone()`s it 55 times, for 56 planes total (28 K + 28 V), 421 MB of fresh writes
   (`attn.rs:184-189, 988-1056`).
3. the prefill's traced greedy head `head_greedy_state` (5.886 ms), the final dense
   `lm_head_fixed_sched` (untraced), `argmax_fixed` and the logits checksum;
4. `cache.insert` — `KvState::clone()` of the 421 MB state, 56 plane clones.

Only the total is measured: **420.759 ms**, of which the traced greedy head is **5.886 ms (1.40%)**;
**414.873 ms is unsplit** across `decode_state`, the dense head, `cache.insert`, the probes,
argmax/checksum and drops.
The earlier draft's `~379 / ~30 / ~11 ms` split is withdrawn: cold `decode_state` (one allocation plus
55 clones) and cold `cache.insert` (56 clones) run the same 421 MB clone loop before any 421 MB free
exists, so the warm/shared turns — which run 2 clones + 2 heads + 1 eviction each — cannot separate
them. The dense head's cost is unmeasured and is not bounded by the bitplane head's 5.886 ms. Two
`Instant`s around `decode_state` and `insert` are the next measurement.

llama-bench's `pp1827` row times `test_prompt` → `llama_decode` + `llama_synchronize`
(`llama-bench.cpp:2093-2120`; the timer is at `:2399-2429`) with the context created at `:2293`; the
KV cache is allocated before the timed path. The allocation, cache clone, probes/drops, argmax/checksum
and the extra greedy head have **no counterpart in the pp row**; the final dense head does correspond
to llama's one head, so the pass-vs-pp comparison is like-for-like on the KV writes, and the
non-counterpart part is a first-turn T2 cost, not a kernel deficit. One asymmetry is named: the cold turn runs the
greedy head (`head_greedy_state`, traced at 5.886 ms) in addition to the final dense head; llama's pp
row computes one head, so the **5.886 ms is the ours-only head, 1.40% of the 0.421 s**.

## What this changes

1. **Target the wait, not the dot.** `pool.wait` is 304 ms turn-wide, **≤7.05% of the pass** if the
   whole wait is charged there; the pass ceiling is `1/(1 − 0.0705) = 1.076×` (upper bound: 5 of the
   1,295 calls sit in the heads, and the turn share is 6.42%), the turn ceiling
   `1/(1 − 0.0642) = 1.069×`. The next change measures a per-dispatch wait histogram (max and p95,
   tagged by the enclosing phase) before touching the pool: the total does not say whether it is one
   phase, one straggler, or 1,290 small skews.
2. **Balance the NQ band.** 913 → 8 × 114/115 instead of 7 × 120 + 73; the seven heavy tasks carry
   4.90% extra, ≤120.5 ms, and the integer split caps the saving at ≤102.55 ms. The tail's 39.16%
   spin is consistent with the 39.17% unused band (whole-window spin, so the match is consistent, not
   exact). Pass ceiling from the loose bound alone `1/(1 − 0.0279) = 1.029×`. This is a work saving,
   additive to the wait only in the model; measure after the wait histogram.
3. **Remove the cold allocation from the turn or make it one buffer.** The 0.421 s is 55.3% of the
   traced gap; the turn ceiling from it alone is `1/(1 − 0.0888) = 1.097×`. A single 421 MB zeroed
   plane sliced into K/V per layer, and a cache `insert` that takes ownership instead of cloning,
   remove the 55 clone writes and the second 421 MB copy. This is T2, not prefill.
4. **The projection is the beyond-parity lever, on `model` evidence.** 44.09 G vs llama's 67.86 G
   (`model`) does not locate the deficit in the projection symbols, but it does not exonerate the
   dot: 57.0% of the pass and the 3.93 s floor bound any win past parity.

The three terms are not summed into a predicted wall: the wait is idle, the band is work, the
outside is allocation, and their fixes interact.

## Withdrawals

- The earlier draft's `+2.00 G → 0.50 s serial glue` and its `0.67 s` closure are withdrawn; see the
  off-CPU identity above.
- The earlier draft's `0.69 s` gap and its llama `454.10 / 463.15 t/s` are not this window; the
  graded pair here is 3.952/3.982/3.977 s.
- The earlier draft's `9.06 G / 6` tail conversion and `0.17 s` are withdrawn; the tail's cycle
  deficit is off-CPU plus band spin, and the band bound is ≤120.5 ms (`model`; integer split
  ≤102.55 ms).
- The outside-pass `~379 / ~30 / ~11 ms` split is withdrawn; only the 420.759 ms total and the
  5.886 ms traced head are measured, and 414.873 ms is unsplit (see above).
- The earlier draft's cycle ledger (double-counted llama pool row, `cycles:u` shares on all-ring
  totals) is replaced by the narrowed projection-symbol comparison above.
- The timestamp join (59.1% unresolved pairs) is not evidence; `pool.wait` replaces it.
- `"The projection is not the problem"` is narrowed to "the projection symbols are cheaper (`model`)
  and the pass work side is **≥ +0.92% above** llama's total wall (`model`); the projection still owns
  57% of the pass wall, and the dot is not exonerated".

## Commands

```sh
# pair (under ~/bin/adg run --), 3 interleaved rounds after load < 1.5
llama-bench -m Qwen3-0.6B-Base.f16.gguf -p 1827 -n 0 -t 8 -r 1
prefix_solve.cleanfin --model skeleton-q8head.gguf --pack nq06-v3.nqp \
  --tokenizer tokenizer.json --len 1827 --differ 0 --decode 0 --mb 440
# phase split (same window)
prefix_solve.cleanfin ... --phases
# per-thread: SIGSTOP at the "len 1827" banner, attach, SIGCONT
perf stat -p $pid --per-thread -e cycles,instructions,task-clock -x, -o t1.csv
perf stat -p $pid --per-thread -e cycles,instructions,branches,branch-misses -x, -o s1.csv
# idle loop: same attach, cycles:u profile
perf record -q -F 4000 -e cycles:u -p $pid -o r1.data
perf report -i r1.data --no-children --stdio --sort symbol
perf report -i r1.data --no-children --stdio --sort pid,symbol
perf annotate -i r1.data -s '…chart4pool6worker' --stdio
```

## Review

`gemini-4-neon-xhigh` ran seven times on this receipt. Pass 1, on the earlier draft: **FAIL, 18
refusals** — the `2.00/3.65` arithmetic and the off-CPU category error (withdrawn, identity above);
the pass/outside conflation; the tail `/8` and spin-diluted IPC; the ledger double count; the
projection overclaim; the timestamp join; the `nq.rs` 7×120+73 cause; the missing artifacts; the
grades; the provenance; the "prefill-only" label; the run count; the unpaired overhead verdict; and
`--phases` not `APEIRON_TRACE`. Pass 2: **FAIL, 13 refusals** — `3,976.650`/`340.591`/`761.350`/
`+36.437` (corrected to `3,976.667`/`340.574`/`761.333`/`+36.420`), `attn.core` 35.0%, the `0.9%
below` sign, the `15.48%` double-multiply in the log, the `pool.wait` scope, the wait-attribution
overclaim and `"of which"`, the outside-pass `~379/~30/~11` split, the missing raw `.log` tables,
the mixed `3.7 ms` window, the `252.6 G` unit and `0.98×`, the epilogue parenthetical, the tail
`dominant` wording, and the spread/citation/load slips. Pass 3: **FAIL, 8 refusals** — the surviving
`"of which"` and turn-wide wait in the pass scope, the `"no counterpart"` overreach for the dense
head, the unbacked `t1` warm/shared walls, the missing `≤` qualifiers, the `≥`/`model` framing, the
llama-wait location, the `2.82%` unit, and the probe order/`KvState` citation. Pass 4: **FAIL, 3
refusals** — the section heading still nested the turn-wide wait in the pass, the 235 µs quotient
mixed the 1,290-call period with the 1,295-call mean, and the sentence "the projection is not where
the wait sits" was an unbacked location claim. Pass 5: **FAIL, 2 refusals** — `5.132 s` was called
the process wall (it is the sum of the printed turn walls; `t1.csv`'s CPUs utilized puts the
attach-to-exit wall at 6.03 s), and "it is not one bad dispatch" contradicted the unmeasured
per-dispatch distribution. Pass 6: **FAIL, 1 refusal** — the `## Review` intro still said the reviewer
ran three times against five recorded passes; the `0.117 s` log line and the argmax/checksum recap
were also aligned. Pass 7: **PASS, 0 refusals** (the pass count, the `≥ 0.117 s` log line, the
6.03 s attach-to-exit wall and the argmax/checksum recap resolve).

A review is critique, not evidence; every accepted claim above resolves to a line in the `.log`.

## Omissions

- Box only; the laptop is not measured, so no speed claim names the target device.
- The phase split is one traced run, 1.0% above the pair median (+47 ms, larger than the 36 ms
  residual); the pair is three runs. The outside 0.421 s is from the trace run, not the pairs.
- The outside-pass contents are unsplit: only the 420.759 ms total and the 5.886 ms traced head are
  measured; the `decode_state`, dense-head, `insert`, argmax/checksum and probe costs have no timer in
the tree.
- The 235 µs per dispatch is a turn-wide total/calls quotient (1,290 in pass + 5 in the heads); the
  distribution and enclosing phase are not measured. That histogram is the next change.
- llama's barrier wait is unmeasured: the `model` `cy.llama.data` table has 5.65% unresolved at
  `0x258a2` and 4.20% unattributed, its event (`cycles` vs `cycles:u`) was not retained, and off-CPU
  sleep does not appear in `cycles:u`. The 0.304 s is our wait, and its agreement with the 340.574 ms
  gap to 36 ms is a consistency check, not a proof that llama has no wait.
- The projection cycle comparison is `model` (different windows/totals); no paired `cycles:u`
  capture of both binaries in one window was made. llama's per-symbol table came from a
  `cy.llama.data` capture for `1795664197` whose event, command and raw file were not retained; the
  `.log` labels the shares `model`.
- The pass is three `compose_cols` calls (913 + 913 + 1 columns). The single-column final step
  re-reads every weight for one token; it sits inside the pass and is not isolated from the
  ≥ +36.420 ms (`model`) pass-minus-wait residual against llama's total wall. Llama-bench splits
  1,827 tokens into four physical ubatches at the default `-ub 512` (512 + 512 + 512 + 291,
  `llama-bench.cpp:376-377`, `llama-context.cpp:1738-1790`) but has no trailing `k = 1` pass.
- The branch misses are per-thread counts from `s1`; no per-thread miss profile. The per-thread and
  profile windows are post-banner lifetime, not the cold pass, and their cold walls differ
  (`t1` 4.862 s, `t2` 4.852 s, `s1` 4.869 s, `s2` 4.882 s, `r1` 5.084 s, `r2` 5.180 s); each table
  names its own window.
- No code, so no cargo/clippy/fmt gate; no Lean, Aeneas, Charon, `--all-features`, GPU or
  `mdbook test`. `check-claims.py` does not scan `bench/cpu/*.md`; `book-check.py` passed. The code
  gates remain as recorded in `1795667807-nq-swiglu-q32-branch.md`.
- Worktree: this receipt and its `.log` are the only new files.
